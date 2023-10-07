#!/bin/bash

pretrained_path=$1
pretrained_type=consolidated
llama_config="$2"
tokenizer_path="$3"
data_config=configs/data/finetune/sg/dialog_chineseSft.yaml

data_parallel=sdp
model_parallel=8

exp_name=finetune/sg/dialog_flacunaChineseSft_70B
echo "exp name: $exp_name"
mkdir -p output/"$exp_name"

# use slurm to launch
python -u main_finetune.py \
--output_dir output/"$exp_name" --epochs 2 --warmup_epochs 0.04 \
--batch_size 2 --accum_iter 64 --num_workers 4 \
--max_words 2048 \
--lr 0.00002 --min_lr 0.0 --clip_grad 2 --weight_decay 0.0 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" --checkpointing \
--llama_type llama --llama_config $llama_config --tokenizer_path "$tokenizer_path" \
--no_visual \
--pretrained_path "$pretrained_path" --pretrained_type="$pretrained_type" \
--data_config $data_config --dialog \
2>&1 | tee -a output/"$exp_name"/output.log

echo "exp name: $exp_name"