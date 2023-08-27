#!/bin/bash

llama_config="$1"
tokenizer_path="$2"
data_meta_path="$3"
data_root="$4"

data_parallel=fsdp
model_parallel=2


exp_name="pretrain/13B_packed"
echo "exp name: $exp_name"
mkdir -p output/"$exp_name"

echo master_addr="$MASTER_ADDR" master_port="$MASTER_PORT" nnodes="$WORLD_SIZE" node_rank="$RANK"

torchrun --nproc_per_node=8 --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
--nnodes="$WORLD_SIZE" --node_rank="$RANK" main_pretrain.py \
--output_dir output/"$exp_name" \
--batch_size 4 --accum_iter 16 --num_workers 4 \
--lr 0.0001 --min_lr 0.00001 --warmup_iters 5000 --lr_decay_iters 400000 --clip_grad 2 --weight_decay 0.02 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" \
--llama_type llama --llama_config "$llama_config" --tokenizer_path "$tokenizer_path" \
--data_meta_path "$data_meta_path" --data_root "$data_root" --packed_data \
2>&1 | tee -a output/"$exp_name"/output"$RANK".log

echo "exp name: $exp_name"
