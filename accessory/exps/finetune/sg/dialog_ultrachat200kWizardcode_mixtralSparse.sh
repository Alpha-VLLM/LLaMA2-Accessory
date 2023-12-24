#!/bin/bash

pretrained_path=$1
pretrained_type=consolidated
llama_config="$2"
tokenizer_path="$3"
data_config=configs/data/finetune/sg/dialog_ultrachat200kWizardcode.yaml

data_parallel=sdp
model_parallel=8

exp_name=finetune/sg/dialog_ultrachat200kWizardcode_mixtralSparse
echo "exp name: $exp_name"
mkdir -p output/"$exp_name"

python -u main_finetune.py \
--output_dir output/"$exp_name" --epochs 1 --warmup_epochs 0.1 \
--batch_size 4 --accum_iter 8 --num_workers 2 \
--max_words 4096 \
--lr 0.000005 --min_lr 0.0 --clip_grad 2 --weight_decay 0.0 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" --checkpointing \
--llama_type mixtral_sparse --llama_config $llama_config --tokenizer_path "$tokenizer_path" \
--no_visual \
--pretrained_path "$pretrained_path" --pretrained_type="$pretrained_type" \
--data_config $data_config --dialog \
2>&1 | tee -a output/"$exp_name"/output.log

echo "exp name: $exp_name"