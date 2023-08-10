#!/bin/bash
# consolidated, meta_ori

# pretrained_path="../checkpoints/Enderfga/"
# pretrained_type=consolidated
pretrained_path="../checkpoints/llama2/Llama-2-70b/"
pretrained_type=meta_ori

llama_config="../checkpoints/Enderfga/params.json"
tokenizer_path="../checkpoints/Enderfga/tokenizer.model"
data_config=configs/data/finetune/sg/alpaca.yaml

data_parallel=sdp
model_parallel=1

exp_name=finetune/sg/alpaca_llamaPeft_normBias_70B
echo "exp name: $exp_name"
mkdir -p output/"$exp_name"

torchrun --master_port=1113 --nproc_per_node=1 main_finetune.py \
--output_dir output/"$exp_name" --epochs 4 --warmup_epochs 1 \
--batch_size 1 --accum_iter 2 --num_workers 4 \
--max_words 512 \
--lr 0.00005 --min_lr 0.000005 --clip_grad 2 --weight_decay 0.02 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" --checkpointing \
--llama_type llama_peft --llama_config "$llama_config" --tokenizer_path "$tokenizer_path" \
--no_visual \
--pretrained_path "$pretrained_path" --pretrained_type="$pretrained_type" \
--data_config $data_config \
--quant \
2>&1 | tee -a output/"$exp_name"/output.log

echo "exp name: $exp_name"