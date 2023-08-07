import argparse
import functools
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

# check that we have a late enough version of transformer
try:
    import transformers
except ImportError:
    raise NotImplementedError("transformers must be installed before converting the weights.")
hf_major_ver, hf_minor_ver = [int(value) for value in transformers.__version__.split(".")[:2]]
if (hf_major_ver, hf_minor_ver) < (4, 31):
    raise NotImplementedError("Requires transformers >= 4.31.0 to convert the weights.")
_format_fn_patterns = {
    "meta_ori": re.compile("^consolidated.\d{2}.pth$"),
    "consolidated": re.compile("^consolidated.\d{2}-of-\d{2}.model.pth$"),
    "consolidated_diff": re.compile("^consolidated.\d{2}-of-\d{2}.model-diff.pth$"),
}


def infer_src_weights_mp_size_and_format(path: str) -> Tuple[int, str]:    
    files_in_folder = os.listdir(path)
    files_in_folder = [fn for fn in files_in_folder if os.path.isfile(os.path.join(path, fn))]
    format_fn_matches = {
        key: [fn for fn in files_in_folder if pattern.match(fn)]
        for key, pattern in _format_fn_patterns.items()
    }
    inferred_format, inferred_mp_size = None, None
    for format, matched_fns in format_fn_matches.items():
        if len(matched_fns) > 0:
            if inferred_format is None:  # first match
                inferred_format = format
                inferred_mp_size = len(matched_fns)
            else:  # multiple matches
                inferred_format = None
                inferred_mp_size = None
                break
    if inferred_format is None or inferred_mp_size is None:
        raise NotImplementedError(
            f"Cannot infer the format of source weights. Files in the folder: {files_in_folder}."
        )
    return inferred_mp_size, inferred_format


def load_src_ckpts(
    format: str, mp_size: int, path: str, meta_pretrained_path: Optional[str], dtype: torch.dtype
) -> List[Dict[str, torch.Tensor]]:
    meta_ori_fns = [
        f"consolidated.{mp_rank:02d}.pth"
        for mp_rank in range(mp_size)
    ]
    consolidated_fns = [
        f"consolidated.{mp_rank:02d}-of-{mp_size:02d}.model.pth"
        for mp_rank in range(mp_size)
    ]
    consolidated_diff_fns = [
        f"consolidated.{mp_rank:02d}-of-{mp_size:02d}.model-diff.pth"
        for mp_rank in range(mp_size)
    ]

    def ckpt_strip_llma_prefix(ckpt: Dict[str, torch.Tensor], mp_rank: int) -> Dict[str, torch.Tensor]:
        prefix = "llma."
        ckpt_strip_prefix = {}
        for key, value in ckpt.items():
            if not key.startswith(prefix):
                print(f"WARNING! Ignoring weight \"key\" due to prefix: "
                      f"Not starting with \"{prefix}\" (at rank {mp_rank}).")
                continue
            ckpt_strip_prefix[key[len(prefix):]] = value
        return ckpt_strip_prefix
    
    def convert_ckpt_dtype(ckpt: Dict[str, torch.Tensor]):
        for key in list(ckpt.keys()):
            ckpt[key] = ckpt[key].to(dtype)

    if format == "meta_ori":
        ckpts = []
        for mp_rank in range(mp_size):
            print(f"Loading shard {mp_rank} of {mp_size} ...")
            ckpts.append(torch.load(os.path.join(path, meta_ori_fns[mp_rank]), map_location="cpu"))
            convert_ckpt_dtype(ckpts[mp_rank])

    elif format == "consolidated":
        ckpts = []
        for mp_rank in range(mp_size):
            print(f"Loading shard {mp_rank} of {mp_size} ...")
            ckpts.append(torch.load(os.path.join(path, consolidated_fns[mp_rank]), map_location="cpu"))
            if "model" in ckpts[mp_rank] and isinstance(ckpts[mp_rank]["model"], dict):
                ckpts[mp_rank] = ckpts[mp_rank]["model"]
            ckpts[mp_rank] = ckpt_strip_llma_prefix(ckpts[mp_rank], mp_rank)
            convert_ckpt_dtype(ckpts[mp_rank])

    elif format == "consolidated_diff":
        meta_pretrained_mp_size = len([
            fn for fn in os.listdir(meta_pretrained_path)
            if _format_fn_patterns["meta_ori"].match(fn) and os.path.isfile(os.path.join(meta_pretrained_path, fn))
        ])
        if meta_pretrained_mp_size != mp_size:
            print("Merging base and diff of different model parallel sizes is not yet supported.")
        merged_ckpt_all = []
        for mp_rank, (fn_diff, fn_base) in enumerate(zip(consolidated_diff_fns, meta_ori_fns)):
            print(f"Loading and combining shard {mp_rank} of {mp_size} ...")
            ckpt_diff = torch.load(os.path.join(path, fn_diff), map_location="cpu")
            ckpt_base = torch.load(os.path.join(meta_pretrained_path, fn_base), map_location="cpu")
            convert_ckpt_dtype(ckpt_diff)
            convert_ckpt_dtype(ckpt_base)
            if "model" in ckpt_diff and isinstance(ckpt_diff["model"], dict):
                ckpt_diff = ckpt_diff["model"]
            ckpt_diff = ckpt_strip_llma_prefix(ckpt_diff, mp_rank)
            merged_ckpt = {}
            for key in ckpt_diff.keys() | ckpt_base.keys():
                if key in ckpt_diff and ckpt_base:
                    merged_ckpt[key] = ckpt_diff[key] + ckpt_base[key]
                    del ckpt_diff[key], ckpt_base[key]
                elif key in ckpt_diff:
                    merged_ckpt[key] = ckpt_diff[key]
                elif key in ckpt_base:
                    merged_ckpt[key] = ckpt_base[key]
            merged_ckpt_all.append(merged_ckpt)
        ckpts = merged_ckpt_all

    return ckpts


def calculate_inv_freq(base: int, head_dim: int) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))


def convert_merged_ckpt_to_hf(
    merged_state_dict: Dict[str, torch.Tensor], params: Dict[str, Any]
) -> List[Dict[str, torch.Tensor]]:
    merged_state_dict = merged_state_dict.copy()
    num_layers = 0
    while f"layers.{num_layers}.attention_norm.weight" in merged_state_dict:
        num_layers += 1
    hf_ckpts = []
    for i in range(num_layers):
        hf_ckpt_shard = {}
        for src_key, dst_key in [
            ("attention.wq.weight", "self_attn.q_proj.weight"),
            ("attention.wk.weight", "self_attn.k_proj.weight"),
            ("attention.wv.weight", "self_attn.v_proj.weight"),
            ("attention.wo.weight", "self_attn.o_proj.weight"),
            ("feed_forward.w1.weight", "mlp.up_proj.weight"),
            ("feed_forward.w2.weight", "mlp.down_proj.weight"),
            ("feed_forward.w3.weight", "mlp.gate_proj.weight"),
            ("attention_norm.weight", "input_layernorm.weight"),
            ("ffn_norm.weight", "post_attention_layernorm.weight"),
        ]:
            dst_key = f"model.layers.{i}." + dst_key
            src_key = f"layers.{i}." + src_key
            hf_ckpt_shard[dst_key] = merged_state_dict[src_key]
            del merged_state_dict[src_key]
        hf_ckpt_shard[f"model.layers.{i}.self_attn.rotary_emb.inv_freq"] = calculate_inv_freq(
            base=10000, head_dim=params["dim"] // params["n_heads"]
        )
        hf_ckpts.append(hf_ckpt_shard)
    
    hf_ckpts.append({})
    for src_key, dst_key in [
        ("norm.weight", "model.norm.weight"),
        ("output.weight", "lm_head.weight"),
        ("tok_embeddings.weight", "model.embed_tokens.weight"),
    ]:
        hf_ckpts[-1][dst_key] = merged_state_dict[src_key]
        del merged_state_dict[src_key]
    assert len(merged_state_dict) == 0, (
        "Unknown key(s) in the source state dict: " + ", ".join(merged_state_dict.keys())
    )

    return hf_ckpts


def write_model_weights(hf_state_dict: List[Dict[str, torch.Tensor]], dest_dir: str) -> None:
    model_index = {
        "metadata": {"total_size": 0},
        "weight_map": {},
    }
    for shard_id, shard_state_dict in enumerate(hf_state_dict):
        shard_fn = f"pytorch_model-{shard_id + 1:05d}-of-{len(hf_state_dict):05d}.bin"
        print(f"Writing to {shard_fn} ...")
        for key, value in shard_state_dict.items():
            model_index["weight_map"][key] = shard_fn
            model_index["metadata"]["total_size"] += (
                value.numel() * torch.finfo(value.dtype).bits * (2 if value.is_complex() else 1)
            )
        torch.save(shard_state_dict, os.path.join(dest_dir, shard_fn))
    with open(os.path.join(dest_dir, "pytorch_model.bin.index.json"), "w") as f:
        json.dump(model_index, f, indent=2)


def write_tokenizer(tokenizer_path: str, dest_dir: str) -> None:
    # From https://github.com/huggingface/transformers/blob/a6e6b1c622d8d08e2510a82cb6266d7b654f1cbf/src/transformers/models/llama/convert_llama_weights_to_hf.py
    try:
        from transformers import LlamaTokenizerFast
    except ImportError as e:
        print(
            "WARNING! The converted tokenizer will be the `slow` tokenizer. "
            "To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
        )
        LlamaTokenizerFast = None
    from transformers import LlamaTokenizer
    tokenizer_class = LlamaTokenizer if LlamaTokenizerFast is None else LlamaTokenizerFast
    tokenizer = tokenizer_class(tokenizer_path)
    tokenizer.save_pretrained(dest_dir)


def write_configs(params: Dict[str, Any], dtype: torch.dtype, dest_dir: str) -> None:
    def calculate_hidden_dim():
        hidden_dim = params["dim"] * 4
        hidden_dim = int(2 * hidden_dim / 3)
        if "ffn_dim_multiplier" in params:
            hidden_dim = int(hidden_dim * params["ffn_dim_multiplier"])
        multiple_of = params["multiple_of"]
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
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
        "num_key_value_heads": params["n_kv_heads"],
        "pad_token_id": 0,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "torch_dtype": {
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.float32: "float32",
        }[dtype],
        "transformers_version": transformers.__version__,
        "use_cache": True,
        "vocab_size": 32000
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
    hf_state_dict: List[Dict[str, torch.Tensor]], dest_dir: str, tokenizer_path: str, params: Dict[str, Any],
    torch_dtype: torch.dtype
) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    print("Writing model weights ...")
    write_model_weights(hf_state_dict, dest_dir)
    print("Writing tokenizer ...")
    write_tokenizer(tokenizer_path, dest_dir)
    print("Writing configs ...")
    write_configs(params, torch_dtype, dest_dir)


def merge_tensor_parallel_weights(ckpts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # Manually specify merge dim for each weight name suffix because:
    # 1. To avoid creating a model (and then infer the merge dim) to save memory.
    # 2. Only weights actually supported by HuggingFace are listed (e.g., biases are not supported now)
    #    so there won't be a lot of corner cases.
    suffix_to_merge_dim = (
        ("tok_embeddings.weight", 1),
        (".attention.wq.weight", 0),
        (".attention.wk.weight", 0),
        (".attention.wv.weight", 0),
        (".attention.wo.weight", 1),
        (".feed_forward.w1.weight", 0),
        (".feed_forward.w2.weight", 1),
        (".feed_forward.w3.weight", 0),
        ("output.weight", 0),
        ("norm.weight", -1),
    )
    merged_ckpt = {}
    for key in sorted(functools.reduce(lambda x, y: x | y, [ckpt.keys() for ckpt in ckpts])):
        print(f"Merging key: {key}")
        assert all(key in ckpt for ckpt in ckpts)
        for suffix, merge_dim in suffix_to_merge_dim:
            if key.endswith(suffix):
                break
        if not key.endswith(suffix):
            raise NotImplementedError(f"Do not know how to merge weights with key: {key}")
        if merge_dim < 0:
            merged_ckpt[key] = ckpts[0][key]
        else:
            merged_ckpt[key] = torch.cat([ckpt[key] for ckpt in ckpts], dim=merge_dim)
        for ckpt in ckpts:
            del ckpt[key]  # to save memory
    return merged_ckpt


def main() -> None:
    parser = argparse.ArgumentParser("Huggingface Weight Conversion Tool")
    parser.add_argument(
        "--src_weights_path", type=str, required=True,
        help="Path to the fine-tuned checkpoints."
    )
    parser.add_argument(
        "--src_config_path", type=str, required=True,
        help="Path to the model configuration file (in the name of params.json as supplied by Meta)."
    )
    parser.add_argument(
        "--meta_pretrained_path", type=str,
        help="Path to the original LLaMA 1/2 checkpoints as released by Meta. In case --src_weights_path "
             "is a weight diff, this argument must be specified."
    )
    parser.add_argument(
        "--dst_weights_path", type=str, required=True,
        help="Path to the converted checkpoint files in Huggingface format."
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True,
        help="Path to the tokenizer.model file as released by Meta."
    )
    parser.add_argument(
        "--dtype", type=str, choices=["fp16", "bf16", "fp32"], default="bf16",
        help="Data type of the converted checkpoints."
    )
    args = parser.parse_args()

    with open(args.src_config_path) as f:
        params = json.load(f)
    src_weights_mp_size, src_weights_format = infer_src_weights_mp_size_and_format(args.src_weights_path)
    print(f"Inferred src weights format: {src_weights_format}, model parallel size: {src_weights_mp_size}.")
    if src_weights_format.endswith("_diff"):
        assert args.meta_pretrained_path is not None, (
            "The src weights is a diff and --meta_pretrained_path must be provided to obtain a full checkpoint."
        )

    torch_dtype = {
        "fp16": torch.half,
        "bf16": torch.bfloat16,
        "fp32": torch.float,
    }[args.dtype]
    print("Loading source checkpoints ...")
    src_ckpts = load_src_ckpts(
        src_weights_format, src_weights_mp_size,
        args.src_weights_path, args.meta_pretrained_path,
        torch_dtype,
    )
    print("Merging tensor parallel weights ...")
    src_ckpt_merged = merge_tensor_parallel_weights(src_ckpts)
    print("Converting to HuggingFace format ...")
    hf_ckpt = convert_merged_ckpt_to_hf(src_ckpt_merged, params)
    print("Writing HuggingFace checkpoints to disk ...")
    write_hf_ckpt(hf_ckpt, args.dst_weights_path, args.tokenizer_path, params, torch_dtype)
    print("Done!")


if __name__ == "__main__":
    main()