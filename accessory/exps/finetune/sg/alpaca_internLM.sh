#!/bin/bash

pretrained_path=$1
pretrained_type=consolidated
llama_config="$2"
tokenizer_path="$3"
data_config=configs/data/finetune/sg/alpaca.yaml

data_parallel=sdp
model_parallel=1

exp_name=finetune/sg/alpaca_internLM
echo "exp name: $exp_name"
mkdir -p output/"$exp_name"

torchrun --master_port=1112 --nproc_per_node=8 main_finetune.py \
--output_dir output/"$exp_name" --epochs 3 --warmup_epochs 0.04 \
--batch_size 4 --accum_iter 8 --num_workers 4 \
--max_words 512 \
--lr 0.00002 --min_lr 0.0 --clip_grad 2 --weight_decay 0.0 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" --checkpointing \
--llama_type internlm --llama_config "$llama_config" --tokenizer_path "$tokenizer_path" \
--no_visual \
--pretrained_path "$pretrained_path" --pretrained_type="$pretrained_type" \
--data_config $data_config \
2>&1 | tee -a output/"$exp_name"/output.log

echo "exp name: $exp_name"