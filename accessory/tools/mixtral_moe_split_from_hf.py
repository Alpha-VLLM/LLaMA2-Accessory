"""
    Rewrite from 
    - https://huggingface.co/Alpha-VLLM/MoE-Mixtral-7B-8Expert/blob/main/converted/split.py
    - https://huggingface.co/Alpha-VLLM/MoE-Mixtral-7B-8Expert/blob/main/converted_sparse/split_sparse.py, 
    but we split from the huggingface version checkpoint
"""

# meta info for mixtral moe split arch
config_json_data = {
    "dim": 4096,
    "hidden_dim": 14336,
    "head_dim": 128,
    "n_layers": 32,
    "n_heads": 32,
    "n_kv_heads": 8,
    "vocab_size": 32000,
    "norm_eps": 1e-05,
    "rope_theta": 1000000,
    "max_batch_size": 32,
    "max_seq_len": 4096,
    "moe": {
        "num_experts_per_tok": 2,
        "num_experts": 8
    },
    "rope_scaling": None
}

meta_json_data = {
  "llama_type": "mistral"
}
# end of meta info


# mapping huggingface ckpt tensor names to magnet ckpt tensor names
hf_name_to_magnet_name = [
    ["lm_head.weight", "llma.output.weight"],
    ["llma.embed_tokens.weight", "llma.tok_embeddings.weight"],
    ["llma.norm.weight", "llma.norm.weight"]
] + sum([[
    [f"llma.layers.{l}.block_sparse_moe.gate.weight", f"llma.layers.{l}.feed_forward.gate.weight"],
    [f"llma.layers.{l}.input_layernorm.weight", f"llma.layers.{l}.attention_norm.weight"],
    [f"llma.layers.{l}.post_attention_layernorm.weight", f"llma.layers.{l}.ffn_norm.weight"],
    [f"llma.layers.{l}.self_attn.k_proj.weight", f"llma.layers.{l}.attention.wk.weight"],
    [f"llma.layers.{l}.self_attn.q_proj.weight", f"llma.layers.{l}.attention.wq.weight"],
    [f"llma.layers.{l}.self_attn.v_proj.weight", f"llma.layers.{l}.attention.wv.weight"],
    [f"llma.layers.{l}.self_attn.o_proj.weight", f"llma.layers.{l}.attention.wo.weight"]] + 
    [[f"llma.layers.{l}.block_sparse_moe.experts.{e}.w1.weight", f"llma.layers.{l}.feed_forward.experts.{e}.w1.weight"] for e in range(8)] +
    [[f"llma.layers.{l}.block_sparse_moe.experts.{e}.w2.weight", f"llma.layers.{l}.feed_forward.experts.{e}.w2.weight"] for e in range(8)] +
    [[f"llma.layers.{l}.block_sparse_moe.experts.{e}.w3.weight", f"llma.layers.{l}.feed_forward.experts.{e}.w3.weight"] for e in range(8)] for l in range(32)], [])
hf_name_to_magnet_name = {item[0]: item[1] for item in hf_name_to_magnet_name}


weight_parallel_dim = {"llma.tok_embeddings.weight": 1, "llma.layers.0.attention.wq.weight": 0,
                       "llma.layers.0.attention.wq.bias": 0, "llma.layers.0.attention.wk.weight": 0,
                       "llma.layers.0.attention.wk.bias": 0, "llma.layers.0.attention.wv.weight": 0,
                       "llma.layers.0.attention.wv.bias": 0, "llma.layers.0.attention.wo.weight": 1,
                       "llma.layers.1.attention.wq.weight": 0, "llma.layers.1.attention.wq.bias": 0,
                       "llma.layers.1.attention.wk.weight": 0, "llma.layers.1.attention.wk.bias": 0,
                       "llma.layers.1.attention.wv.weight": 0, "llma.layers.1.attention.wv.bias": 0,
                       "llma.layers.1.attention.wo.weight": 1, "llma.layers.2.attention.wq.weight": 0,
                       "llma.layers.2.attention.wq.bias": 0, "llma.layers.2.attention.wk.weight": 0,
                       "llma.layers.2.attention.wk.bias": 0, "llma.layers.2.attention.wv.weight": 0,
                       "llma.layers.2.attention.wv.bias": 0, "llma.layers.2.attention.wo.weight": 1,
                       "llma.layers.3.attention.wq.weight": 0, "llma.layers.3.attention.wq.bias": 0,
                       "llma.layers.3.attention.wk.weight": 0, "llma.layers.3.attention.wk.bias": 0,
                       "llma.layers.3.attention.wv.weight": 0, "llma.layers.3.attention.wv.bias": 0,
                       "llma.layers.3.attention.wo.weight": 1, "llma.layers.4.attention.wq.weight": 0,
                       "llma.layers.4.attention.wq.bias": 0, "llma.layers.4.attention.wk.weight": 0,
                       "llma.layers.4.attention.wk.bias": 0, "llma.layers.4.attention.wv.weight": 0,
                       "llma.layers.4.attention.wv.bias": 0, "llma.layers.4.attention.wo.weight": 1,
                       "llma.layers.5.attention.wq.weight": 0, "llma.layers.5.attention.wq.bias": 0,
                       "llma.layers.5.attention.wk.weight": 0, "llma.layers.5.attention.wk.bias": 0,
                       "llma.layers.5.attention.wv.weight": 0, "llma.layers.5.attention.wv.bias": 0,
                       "llma.layers.5.attention.wo.weight": 1, "llma.layers.6.attention.wq.weight": 0,
                       "llma.layers.6.attention.wq.bias": 0, "llma.layers.6.attention.wk.weight": 0,
                       "llma.layers.6.attention.wk.bias": 0, "llma.layers.6.attention.wv.weight": 0,
                       "llma.layers.6.attention.wv.bias": 0, "llma.layers.6.attention.wo.weight": 1,
                       "llma.layers.7.attention.wq.weight": 0, "llma.layers.7.attention.wq.bias": 0,
                       "llma.layers.7.attention.wk.weight": 0, "llma.layers.7.attention.wk.bias": 0,
                       "llma.layers.7.attention.wv.weight": 0, "llma.layers.7.attention.wv.bias": 0,
                       "llma.layers.7.attention.wo.weight": 1, "llma.layers.8.attention.wq.weight": 0,
                       "llma.layers.8.attention.wq.bias": 0, "llma.layers.8.attention.wk.weight": 0,
                       "llma.layers.8.attention.wk.bias": 0, "llma.layers.8.attention.wv.weight": 0,
                       "llma.layers.8.attention.wv.bias": 0, "llma.layers.8.attention.wo.weight": 1,
                       "llma.layers.9.attention.wq.weight": 0, "llma.layers.9.attention.wq.bias": 0,
                       "llma.layers.9.attention.wk.weight": 0, "llma.layers.9.attention.wk.bias": 0,
                       "llma.layers.9.attention.wv.weight": 0, "llma.layers.9.attention.wv.bias": 0,
                       "llma.layers.9.attention.wo.weight": 1, "llma.layers.10.attention.wq.weight": 0,
                       "llma.layers.10.attention.wq.bias": 0, "llma.layers.10.attention.wk.weight": 0,
                       "llma.layers.10.attention.wk.bias": 0, "llma.layers.10.attention.wv.weight": 0,
                       "llma.layers.10.attention.wv.bias": 0, "llma.layers.10.attention.wo.weight": 1,
                       "llma.layers.11.attention.wq.weight": 0, "llma.layers.11.attention.wq.bias": 0,
                       "llma.layers.11.attention.wk.weight": 0, "llma.layers.11.attention.wk.bias": 0,
                       "llma.layers.11.attention.wv.weight": 0, "llma.layers.11.attention.wv.bias": 0,
                       "llma.layers.11.attention.wo.weight": 1, "llma.layers.12.attention.wq.weight": 0,
                       "llma.layers.12.attention.wq.bias": 0, "llma.layers.12.attention.wk.weight": 0,
                       "llma.layers.12.attention.wk.bias": 0, "llma.layers.12.attention.wv.weight": 0,
                       "llma.layers.12.attention.wv.bias": 0, "llma.layers.12.attention.wo.weight": 1,
                       "llma.layers.13.attention.wq.weight": 0, "llma.layers.13.attention.wq.bias": 0,
                       "llma.layers.13.attention.wk.weight": 0, "llma.layers.13.attention.wk.bias": 0,
                       "llma.layers.13.attention.wv.weight": 0, "llma.layers.13.attention.wv.bias": 0,
                       "llma.layers.13.attention.wo.weight": 1, "llma.layers.14.attention.wq.weight": 0,
                       "llma.layers.14.attention.wq.bias": 0, "llma.layers.14.attention.wk.weight": 0,
                       "llma.layers.14.attention.wk.bias": 0, "llma.layers.14.attention.wv.weight": 0,
                       "llma.layers.14.attention.wv.bias": 0, "llma.layers.14.attention.wo.weight": 1,
                       "llma.layers.15.attention.wq.weight": 0, "llma.layers.15.attention.wq.bias": 0,
                       "llma.layers.15.attention.wk.weight": 0, "llma.layers.15.attention.wk.bias": 0,
                       "llma.layers.15.attention.wv.weight": 0, "llma.layers.15.attention.wv.bias": 0,
                       "llma.layers.15.attention.wo.weight": 1, "llma.layers.16.attention.wq.weight": 0,
                       "llma.layers.16.attention.wq.bias": 0, "llma.layers.16.attention.wk.weight": 0,
                       "llma.layers.16.attention.wk.bias": 0, "llma.layers.16.attention.wv.weight": 0,
                       "llma.layers.16.attention.wv.bias": 0, "llma.layers.16.attention.wo.weight": 1,
                       "llma.layers.17.attention.wq.weight": 0, "llma.layers.17.attention.wq.bias": 0,
                       "llma.layers.17.attention.wk.weight": 0, "llma.layers.17.attention.wk.bias": 0,
                       "llma.layers.17.attention.wv.weight": 0, "llma.layers.17.attention.wv.bias": 0,
                       "llma.layers.17.attention.wo.weight": 1, "llma.layers.18.attention.wq.weight": 0,
                       "llma.layers.18.attention.wq.bias": 0, "llma.layers.18.attention.wk.weight": 0,
                       "llma.layers.18.attention.wk.bias": 0, "llma.layers.18.attention.wv.weight": 0,
                       "llma.layers.18.attention.wv.bias": 0, "llma.layers.18.attention.wo.weight": 1,
                       "llma.layers.19.attention.wq.weight": 0, "llma.layers.19.attention.wq.bias": 0,
                       "llma.layers.19.attention.wk.weight": 0, "llma.layers.19.attention.wk.bias": 0,
                       "llma.layers.19.attention.wv.weight": 0, "llma.layers.19.attention.wv.bias": 0,
                       "llma.layers.19.attention.wo.weight": 1, "llma.layers.20.attention.wq.weight": 0,
                       "llma.layers.20.attention.wq.bias": 0, "llma.layers.20.attention.wk.weight": 0,
                       "llma.layers.20.attention.wk.bias": 0, "llma.layers.20.attention.wv.weight": 0,
                       "llma.layers.20.attention.wv.bias": 0, "llma.layers.20.attention.wo.weight": 1,
                       "llma.layers.21.attention.wq.weight": 0, "llma.layers.21.attention.wq.bias": 0,
                       "llma.layers.21.attention.wk.weight": 0, "llma.layers.21.attention.wk.bias": 0,
                       "llma.layers.21.attention.wv.weight": 0, "llma.layers.21.attention.wv.bias": 0,
                       "llma.layers.21.attention.wo.weight": 1, "llma.layers.22.attention.wq.weight": 0,
                       "llma.layers.22.attention.wq.bias": 0, "llma.layers.22.attention.wk.weight": 0,
                       "llma.layers.22.attention.wk.bias": 0, "llma.layers.22.attention.wv.weight": 0,
                       "llma.layers.22.attention.wv.bias": 0, "llma.layers.22.attention.wo.weight": 1,
                       "llma.layers.23.attention.wq.weight": 0, "llma.layers.23.attention.wq.bias": 0,
                       "llma.layers.23.attention.wk.weight": 0, "llma.layers.23.attention.wk.bias": 0,
                       "llma.layers.23.attention.wv.weight": 0, "llma.layers.23.attention.wv.bias": 0,
                       "llma.layers.23.attention.wo.weight": 1, "llma.layers.24.attention.wq.weight": 0,
                       "llma.layers.24.attention.wq.bias": 0, "llma.layers.24.attention.wk.weight": 0,
                       "llma.layers.24.attention.wk.bias": 0, "llma.layers.24.attention.wv.weight": 0,
                       "llma.layers.24.attention.wv.bias": 0, "llma.layers.24.attention.wo.weight": 1,
                       "llma.layers.25.attention.wq.weight": 0, "llma.layers.25.attention.wq.bias": 0,
                       "llma.layers.25.attention.wk.weight": 0, "llma.layers.25.attention.wk.bias": 0,
                       "llma.layers.25.attention.wv.weight": 0, "llma.layers.25.attention.wv.bias": 0,
                       "llma.layers.25.attention.wo.weight": 1, "llma.layers.26.attention.wq.weight": 0,
                       "llma.layers.26.attention.wq.bias": 0, "llma.layers.26.attention.wk.weight": 0,
                       "llma.layers.26.attention.wk.bias": 0, "llma.layers.26.attention.wv.weight": 0,
                       "llma.layers.26.attention.wv.bias": 0, "llma.layers.26.attention.wo.weight": 1,
                       "llma.layers.27.attention.wq.weight": 0, "llma.layers.27.attention.wq.bias": 0,
                       "llma.layers.27.attention.wk.weight": 0, "llma.layers.27.attention.wk.bias": 0,
                       "llma.layers.27.attention.wv.weight": 0, "llma.layers.27.attention.wv.bias": 0,
                       "llma.layers.27.attention.wo.weight": 1, "llma.layers.28.attention.wq.weight": 0,
                       "llma.layers.28.attention.wq.bias": 0, "llma.layers.28.attention.wk.weight": 0,
                       "llma.layers.28.attention.wk.bias": 0, "llma.layers.28.attention.wv.weight": 0,
                       "llma.layers.28.attention.wv.bias": 0, "llma.layers.28.attention.wo.weight": 1,
                       "llma.layers.29.attention.wq.weight": 0, "llma.layers.29.attention.wq.bias": 0,
                       "llma.layers.29.attention.wk.weight": 0, "llma.layers.29.attention.wk.bias": 0,
                       "llma.layers.29.attention.wv.weight": 0, "llma.layers.29.attention.wv.bias": 0,
                       "llma.layers.29.attention.wo.weight": 1, "llma.layers.30.attention.wq.weight": 0,
                       "llma.layers.30.attention.wq.bias": 0, "llma.layers.30.attention.wk.weight": 0,
                       "llma.layers.30.attention.wk.bias": 0, "llma.layers.30.attention.wv.weight": 0,
                       "llma.layers.30.attention.wv.bias": 0, "llma.layers.30.attention.wo.weight": 1,
                       "llma.layers.31.attention.wq.weight": 0, "llma.layers.31.attention.wq.bias": 0,
                       "llma.layers.31.attention.wk.weight": 0, "llma.layers.31.attention.wk.bias": 0,
                       "llma.layers.31.attention.wv.weight": 0, "llma.layers.31.attention.wv.bias": 0,
                       "llma.layers.31.attention.wo.weight": 1, "llma.output.weight": 0, "llma.output.bias": 0}

import argparse
import torch
from pathlib import Path
import shutil
import json

parser = argparse.ArgumentParser()
parser.add_argument('in_folder', type=str, help='Model folder that stores the original ckpt')
parser.add_argument('out_folder', type=str, help='Model folder that stores the output ckpt')
parser.add_argument('--in_ckpt_source', type=str, default='hf', choices=['hf', 'magnet'], help='Input model folder source')
parser.add_argument('--convert_sparse', action='store_true', help='Convert to the sparse format')
if __name__ == '__main__':
    args = parser.parse_args()

    Path(args.out_folder).mkdir(exist_ok=True)

    # save misc other things
    shutil.copy(Path(args.in_folder) / 'tokenizer.model', Path(args.out_folder) / 'tokenizer.model')
    with open(Path(args.out_folder) / 'meta.json', 'w') as f:
        json.dump(meta_json_data, f)
    with open(Path(args.out_folder) / 'config.json', 'w') as f:
        json.dump(config_json_data, f)

    if args.in_ckpt_source == 'magnet':
        ori = torch.load("consolidated.00.pth", map_location="cpu")
        ori = {"llma." + key: val for key, val in ori.items()}
    else:
        ori = {}

        import json
        import os.path as osp
        import safetensors
        from safetensors import safe_open

        with open(osp.join(args.in_folder, 'model.safetensors.index.json'), 'r') as f:
            the_map = json.load(f)
        print('metadata:', the_map['metadata'])
        all_partitions = set(the_map['weight_map'].values())
        for now_partition in all_partitions:
            with safe_open(osp.join(args.in_folder, now_partition), framework="pt", device="cpu") as f:
                for key in f.keys():
                    new_key = hf_name_to_magnet_name[key.replace('model.', 'llma.')]
                    ori[new_key] = f.get_tensor(key)

                    if "wq" in new_key or "wk" in new_key:
                        print('transposing', new_key)
                        # to be compatible with HuggingFace's pos embed implementation.
                        head_dim = 128
                        in_dim = ori[new_key].size(1)
                        ori[new_key] = ori[new_key].view(
                            -1, 2, head_dim // 2, in_dim,
                        ).transpose(1, 2).flatten(0, 2).contiguous()

    def func(rank=0):
        shard_split_to = 8
        split_ckpt = {}
        for key, ori_param in ori.items():
            if key in weight_parallel_dim:
                split_ckpt[key] = torch.chunk(ori_param, shard_split_to, weight_parallel_dim[key])[
                    rank % shard_split_to].clone()
                if args.in_ckpt_source == 'hf':
                    split_ckpt[key] = split_ckpt[key].half()
                if rank == 0:
                    print(f"chunk {key}")
            else:
                if not args.convert_sparse:
                    if "experts." in key and int(key.split("experts.")[1].split(".")[0]) != rank:
                        continue
                    else:
                        split_ckpt[key] = ori_param
                        if args.in_ckpt_source == 'hf':
                            split_ckpt[key] = split_ckpt[key].half()
                        if rank == 0:
                            print(f"inherit {key}")
                else:
                    if "experts.0." in key:
                        weight_all_experts = [ori[key.replace("experts.0.", f"experts.{i}.")] for i in range(8)]
                        if "w2" in key:
                            weight_all_experts = [torch.transpose(_, 0, 1) for _ in weight_all_experts]
                        weight_this_rank = [torch.chunk(_, 8, dim=0)[rank] for _ in weight_all_experts]
                        weight_this_rank = torch.cat(weight_this_rank, dim=0).clone()
                        key = key.replace("experts.0.", "").replace(".weight", "")
                        split_ckpt[key] = weight_this_rank
                        if args.in_ckpt_source == 'hf':
                            split_ckpt[key] = split_ckpt[key].half()
                        print("expert key")
                    elif "experts" in key:
                        continue
                    else:
                        split_ckpt[key] = ori_param
                        if args.in_ckpt_source == 'hf':
                            split_ckpt[key] = split_ckpt[key].half()
                        if rank == 0:
                            print(f"inherit {key}")
        print('saving at rank', rank)
        torch.save({"model": split_ckpt}, osp.join(args.out_folder, f"consolidated.{rank:02d}-of-08.model.pth"))

    for r in range(8):
        func(r)
    
