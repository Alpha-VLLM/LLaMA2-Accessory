#!/bin/bash

pretrained_path=$1
pretrained_type=consolidated
llama_config="$2 configs/model/finetune/sg/llamaPeft_normBiasLora.json"
tokenizer_path="$3"
data_config=configs/data/finetune/mm/alpaca_llava.yaml

data_parallel=sdp
model_parallel=2

exp_name=finetune/mm/alpacaLlava_llamaQformerv2Peft_13B
echo "exp name: $exp_name"
mkdir -p output/"$exp_name"

torchrun --master_port=1112 --nproc_per_node=8 main_finetune.py \
--output_dir output/"$exp_name" --epochs 3 --warmup_epochs 0.2 \
--batch_size 16 --accum_iter 2 --num_workers 4 \
--max_words 512 \
--lr 0.00005 --min_lr 0.000005 --clip_grad 2 --weight_decay 0.02 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" --checkpointing \
--llama_type llama_qformerv2_peft --llama_config $llama_config --tokenizer_path "$tokenizer_path" \
--pretrained_path "$pretrained_path" --pretrained_type="$pretrained_type" \
--data_config $data_config \
2>&1 | tee -a output/"$exp_name"/output.log

echo "exp name: $exp_name"