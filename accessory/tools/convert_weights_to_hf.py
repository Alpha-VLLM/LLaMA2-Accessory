r"""This tool converts a trained checkpoint in our tensor-parallel format to
the huggingface format.

The tool also supports merging the weights: If you have multiple checkpoints
(e.g., you have a set of base weights from Meta, and a set of delta weights
downloaded from our repo), you can pass them all to the ``--src_weights_path``
argument. Delta weights should come AFTER the base weights.

The tool only supports converting models that are supported by HuggingFace
(e.g., bias-tuning models are not supported since ``bias=False`` is hard-coded
in its Linear layers at this moment).

The tool requires that HuggingFace transformers >= 4.31 be installed.

The tool consumes considerable amount of CPU RAM: For 7B in FP16 we recommend
32GB of CPU RAM and for 70B we recommend 192GB. If your machine do not have
that much RAM, you can work around by allocating some swap memory, but it may
make the conversion process much slower.

Example usage::

    # The folders to prepare:
    #
    # /path/to/llama-2-70b: Path to the original LLaMA-2-70B weights by Meta.
    # /path/to/finetune/sg/dialog_sharegpt_70b: ShareGPT finetuned delta
    #   weights downloaded from our repo.
    # /path/to/llama/tokenizer.model: Tokenizer model file released by Meta.
    # /path/to/llama2_accessory_github_repo: Path to the cloned Github repo.
    #
    # Then, run in Bash:

    $ cd /path/to/llama2_accessory_github_repo
    $ python -m tools.convert_weights_to_hf \
        --src_weights_path /path/to/llama-2-70b \
            /path/to/finetune/sg/dialog_sharegpt_70b \
        --src_config_path /path/to/llama-2-70b/params.json \
        --tokenizer_path /path/to/llama/tokenizer.model \
        --dst_weights_path /path/to/llama-2-70b-hf-sharegpt

    # If the model to convert contains unknown parameters (e.g., converting a
    # multi-modal model to huggingface LLaMA which is language-only), add
    # --ignore_unknown_keys to the command above.

    # Then, use in transformers (in Python):
    >>> from transformers import AutoModelForCausalLM
    >>> AutoModelForCausalLM.from_pretrained(
    >>>     "/path/to/llama-2-70b-hf-sharegpt"
    >>> )

Note:

    In some early versions of ``transformers`` (e.g., 4.31.0), you may see some
    error messages saying ``model.layers.*.self_attn.rotary_emb.inv_freq`` is
    missing. This is the expected behavior since ``inv_freq`` has changed to
    non-persistent buffers in the future versions of ``transformers`` (
    https://github.com/huggingface/transformers/pull/24998). We comply with the
    latter model spec. This warning can be ignored since the values in the
    checkpoint should be identical to the newly initialized values. Upgrading
    to the latest version of ``transformers`` (e.g., >= 4.32.0) should get rid
    of the warning.
"""

import argparse
import json
import os
from typing import Any, Dict, List
import re

import torch

from accessory.util.tensor_parallel import (
    infer_checkpoint_format_and_mp_size,
    load_tensor_parallel_shard_state_dict,
    ShardedTensorLoader,
)

# check that we have a late enough version of transformers
try:
    import transformers
except ImportError:
    raise NotImplementedError("transformers must be installed before "
                              "converting the weights.")
print("transformers version:", transformers.__version__)
hf_major_ver, hf_minor_ver = [
    int(value) for value in transformers.__version__.split(".")[:2]
]
if (hf_major_ver, hf_minor_ver) < (4, 31):
    raise NotImplementedError("Requires transformers >= 4.31.0 to convert the "
                              "weights.")


def load_and_merge_tensor_parallel_weights(
    src_weights_path: List[str], torch_dtype: torch.dtype, 
    ignore_unknown_keys: bool = False,
) -> Dict[str, torch.Tensor]:
    # Manually specify merge dim for each weight name pattern because:
    # 1. To avoid creating a model (and then infer the merge dim) to save
    #    memory.
    # 2. Only weights actually supported by HuggingFace are listed (e.g.,
    #    biases are not supported now) so there won't be a lot of corner cases.
    pattern_to_merge_dim = (
        ("^llma.tok_embeddings.weight$", 1),
        ("^llma.layers.(\d+).attention.wq.weight$", 0),
        ("^llma.layers.(\d+).attention.wk.weight$", 0),
        ("^llma.layers.(\d+).attention.wv.weight$", 0),
        ("^llma.layers.(\d+).attention.wo.weight$", 1),
        ("^llma.layers.(\d+).attention_norm.weight", -1),
        ("^llma.layers.(\d+).feed_forward.w1.weight$", 0),
        ("^llma.layers.(\d+).feed_forward.w2.weight$", 1),
        ("^llma.layers.(\d+).feed_forward.w3.weight$", 0),
        ("^llma.layers.(\d+).ffn_norm.weight", -1),
        ("^llma.output.weight$", 0),
        ("^llma.norm.weight$", -1),
        ("^llma.rope.freqs$", -1),
    )
    pattern_to_merge_dim = tuple(
        (re.compile(pattern), dim)
        for pattern, dim in pattern_to_merge_dim
    )
    merged_ckpt = {}
    ignored_keys = []
    for i, path in enumerate(src_weights_path):
        format, mp_size = infer_checkpoint_format_and_mp_size(path)
        sharded_tensor_loaders: Dict[str, ShardedTensorLoader] = {}
        if i == 0:
            assert not format.endswith("_diff"), (
                "A base checkpoint is needed as the first checkpoint, "
                "not a delta checkpoint."
            )
        for shard_id in range(mp_size):
            print(f"Loading shard {shard_id} of {mp_size} of checkpoint: "
                  f"{path} (inferred format is {format})")
            ckpt_shard = load_tensor_parallel_shard_state_dict(
                path, format, shard_id, mp_size
            )
            for key, value in ckpt_shard.items():
                matched = False
                for pattern, merge_dim in pattern_to_merge_dim:
                    if pattern.match(key):
                        matched = True
                        break
                if not matched:
                    if key not in ignored_keys:
                        ignored_keys.append(key)
                    continue
                if key not in merged_ckpt:
                    merged_size = list(value.size())
                    if merge_dim >= 0:
                        merged_size[merge_dim] *= mp_size
                    init_dtype = (
                        torch_dtype
                        if value.is_floating_point() else
                        value.dtype
                    )
                    merged_ckpt[key] = torch.zeros(merged_size,
                                                   dtype=init_dtype)
                if key not in sharded_tensor_loaders:
                    sharded_tensor_loaders[key] = ShardedTensorLoader(
                        merged_ckpt[key], mp_size, merge_dim,
                        mode="add" if format.endswith("_diff") else "set"
                    )
                sharded_tensor_loaders[key].load_shard(shard_id, value)

        for key, value in sharded_tensor_loaders.items():
            assert value.is_complete(), (
                "A key is not loaded completely after going through all "
                "shards. Please check the integrity of the checkpoint folder ("
                f"key: {key}, checkpoint: {path})."
            )

    if len(ignored_keys) > 0:            
        print("Unknown key(s) found in source checkpoint:",
              ", ".join(ignored_keys))
        assert ignore_unknown_keys, (
            "To ignore unknown keys, relaunch with --ignore_unknown_keys "
            "in the command line arguments."
        )

    return merged_ckpt


def convert_merged_ckpt_to_hf(
    merged_state_dict: Dict[str, torch.Tensor], params: Dict[str, Any],
) -> List[Dict[str, torch.Tensor]]:
    merged_state_dict = merged_state_dict.copy()
    num_layers = 0
    while (f"llma.layers.{num_layers}.attention_norm.weight"
           in merged_state_dict):
        num_layers += 1
    hf_ckpts = []
    if "llma.rope.freqs" in merged_state_dict:
        del merged_state_dict["llma.rope.freqs"]
    for i in range(num_layers):
        hf_ckpt_shard = {}
        for src_key, dst_key in [
            ("attention.wq.weight", "self_attn.q_proj.weight"),
            ("attention.wk.weight", "self_attn.k_proj.weight"),
            ("attention.wv.weight", "self_attn.v_proj.weight"),
            ("attention.wo.weight", "self_attn.o_proj.weight"),
            ("feed_forward.w3.weight", "mlp.up_proj.weight"),
            ("feed_forward.w2.weight", "mlp.down_proj.weight"),
            ("feed_forward.w1.weight", "mlp.gate_proj.weight"),
            ("attention_norm.weight", "input_layernorm.weight"),
            ("ffn_norm.weight", "post_attention_layernorm.weight"),
        ]:
            dst_key = f"model.layers.{i}." + dst_key
            src_key = f"llma.layers.{i}." + src_key
            value = merged_state_dict[src_key]
            if "q_proj" in dst_key or "k_proj" in dst_key:
                # to be compatible with HuggingFace's pos embed implementation.
                if "q_proj" in dst_key:
                    n_heads = params["n_heads"]
                else:  # "k_proj" in dst_key:
                    n_heads = params.get("n_kv_heads", params["n_heads"])
                head_dim = value.size(0) // n_heads
                in_dim = value.size(1)
                value = value.view(
                    n_heads, head_dim // 2, 2, in_dim,
                ).transpose(1, 2).flatten(0, 2)
            hf_ckpt_shard[dst_key] = value
            del merged_state_dict[src_key]
        hf_ckpts.append(hf_ckpt_shard)

    hf_ckpts.append({})
    for src_key, dst_key in [
        ("llma.norm.weight", "model.norm.weight"),
        ("llma.output.weight", "lm_head.weight"),
        ("llma.tok_embeddings.weight", "model.embed_tokens.weight"),
    ]:
        hf_ckpts[-1][dst_key] = merged_state_dict[src_key]
        del merged_state_dict[src_key]
    assert len(merged_state_dict) == 0, (
        "Unknown key(s) in the source state dict: "
        + ", ".join(merged_state_dict.keys())
    )

    return hf_ckpts


def write_model_weights(
    hf_state_dict: List[Dict[str, torch.Tensor]], dest_dir: str
) -> None:
    model_index = {
        "metadata": {"total_size": 0},
        "weight_map": {},
    }
    for shard_id, shard_state_dict in enumerate(hf_state_dict):
        shard_fn = (
            f"pytorch_model-{shard_id + 1:05d}-of-"
            f"{len(hf_state_dict):05d}.bin"
        )
        print(f"Writing to {shard_fn} ...")
        for key, value in shard_state_dict.items():
            model_index["weight_map"][key] = shard_fn
            model_index["metadata"]["total_size"] += (
                value.numel() * torch.finfo(value.dtype).bits
                * (2 if value.is_complex() else 1)
            )
        torch.save(shard_state_dict, os.path.join(dest_dir, shard_fn))
    with open(
        os.path.join(dest_dir, "pytorch_model.bin.index.json"), "w"
    ) as f:
        json.dump(model_index, f, indent=2)


def write_tokenizer(tokenizer_path: str, dest_dir: str) -> Any:
    # From https://github.com/huggingface/transformers/blob/a6e6b1c622d8d08e2510a82cb6266d7b654f1cbf/src/transformers/models/llama/convert_llama_weights_to_hf.py  # noqa: E501
    try:
        from transformers import LlamaTokenizerFast
    except ImportError:
        print(
            "WARNING! The converted tokenizer will be the `slow` tokenizer. "
            "To use the fast, update your `tokenizers` library and re-run the "
            "tokenizer conversion."
        )
        LlamaTokenizerFast = None
    from transformers import LlamaTokenizer
    tokenizer_class = LlamaTokenizerFast or LlamaTokenizer
    tokenizer = tokenizer_class(tokenizer_path)
    tokenizer.save_pretrained(dest_dir)
    return tokenizer


def write_configs(
    params: Dict[str, Any], dtype: torch.dtype, dest_dir: str, vocab_size: int
) -> None:
    def calculate_hidden_dim():
        hidden_dim = params["dim"] * 4
        hidden_dim = int(2 * hidden_dim / 3)
        if "ffn_dim_multiplier" in params:
            hidden_dim = int(hidden_dim * params["ffn_dim_multiplier"])
        multiple_of = params["multiple_of"]
        hidden_dim = (
            multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        )
        return hidden_dim

    config = {
        "architectures": [
            "LlamaForCausalLM"
        ],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": params["dim"],
        "initializer_range": 0.02,
        "intermediate_size": calculate_hidden_dim(),
        "max_position_embeddings": 2048,
        "model_type": "llama",
        "num_attention_heads": params["n_heads"],
        "num_hidden_layers": params["n_layers"],
        "num_key_value_heads": params.get("n_kv_heads", params["n_heads"]),
        "pad_token_id": 0,
        "pretraining_tp": 1,
        "rms_norm_eps": params.get("norm_eps", 1e-5),
        "rope_theta": params.get("rope_theta", 10000),
        "rope_scaling": None if "rope_scaling" not in params else {
            "type": "linear",
            "factor": params["rope_scaling"],
        },
        "tie_word_embeddings": False,
        "torch_dtype": {
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.float32: "float32",
        }[dtype],
        "transformers_version": transformers.__version__,
        "use_cache": True,
        "vocab_size": vocab_size
    }
    with open(os.path.join(dest_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    generation_config = {
        "_from_model_config": True,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "transformers_version": transformers.__version__,
    }
    with open(os.path.join(dest_dir, "generation_config.json"), "w") as f:
        json.dump(generation_config, f, indent=2)


def write_hf_ckpt(
    hf_state_dict: List[Dict[str, torch.Tensor]], dest_dir: str,
    tokenizer_path: str, params: Dict[str, Any], torch_dtype: torch.dtype
) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    print("Writing model weights ...")
    write_model_weights(hf_state_dict, dest_dir)
    print("Writing tokenizer ...")
    tokenizer = write_tokenizer(tokenizer_path, dest_dir)
    print("Writing configs ...")
    write_configs(params, torch_dtype, dest_dir, tokenizer.vocab_size)


def main() -> None:
    parser = argparse.ArgumentParser("Huggingface Weight Conversion Tool")
    parser.add_argument(
        "--src_weights_path", type=str, required=True, nargs="+",
        help="Path(s) to the finetuned checkpoints. If multiple checkpoint "
             "folders are provided, they will be merged from the left to the "
             "right."
    )
    parser.add_argument(
        "--src_config_path", type=str, required=True, nargs="+",
        help="Path to the model configuration files (in the name of "
             "params.json as supplied by Meta). Multiple config files are "
             "supported and will be merged from left to right (i.e., the "
             "configuration items in the later files override the previous "
             "ones)."
    )
    parser.add_argument(
        "--dst_weights_path", type=str, required=True,
        help="Path to the converted checkpoint files in HuggingFace format."
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True,
        help="Path to the tokenizer.model file as released by Meta."
    )
    parser.add_argument(
        "--dtype", type=str, choices=["fp16", "bf16", "fp32"], default="bf16",
        help="Data type of the converted checkpoints."
    )
    parser.add_argument(
        "--ignore_unknown_keys", action="store_true",
        help="Ignore unknown keys in the source checkpoint (the scripts will "
             "only give warnings); otherwise the conversion will fail."
    )
    args = parser.parse_args()

    params = {}
    for path in args.src_config_path:
        with open(path) as f:
            params.update(json.load(f))

    torch_dtype = {
        "fp16": torch.half,
        "bf16": torch.bfloat16,
        "fp32": torch.float,
    }[args.dtype]

    print("Loading and merging source checkpoints ...")
    src_ckpt_merged = load_and_merge_tensor_parallel_weights(
        args.src_weights_path, torch_dtype, args.ignore_unknown_keys
    )
    print("Converting to HuggingFace format ...")
    hf_ckpt = convert_merged_ckpt_to_hf(src_ckpt_merged, params)
    print("Writing HuggingFace checkpoints to disk ...")
    write_hf_ckpt(hf_ckpt, args.dst_weights_path, args.tokenizer_path, params,
                  torch_dtype)
    print("Done!")


if __name__ == "__main__":
    main()
