#!/bin/bash

pretrained_path=$1
pretrained_type=consolidated
llama_config="$2"
tokenizer_path="$3"
data_config=configs/data/finetune/mm/dialog_videochatgpt.yaml

data_parallel=sdp
model_parallel=2

exp_name=finetune/mm/dialog_videochatgpt_llamaInterleave_13B
echo "exp name: $exp_name"
mkdir -p output/"$exp_name"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -u main_finetune.py \
--output_dir output/"$exp_name" --epochs 1 --warmup_epochs 0.03 \
--batch_size 4 --accum_iter 4 --num_workers 2 \
--max_words 4096 \
--lr 0.00002 --min_lr 0 --clip_grad 8 --weight_decay 0 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" --checkpointing \
--llama_type llama_interleave --llama_config $llama_config --tokenizer_path "$tokenizer_path" \
--pretrained_path "$pretrained_path" --pretrained_type="$pretrained_type" \
--data_config $data_config --dialog \
--image_transform padded_resize \
2>&1 | tee -a output/"$exp_name"/output.log

echo "exp name: $exp_name"