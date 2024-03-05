#!/usr/bin/env sh

train_data_root='configs/data/JourneyDB.yaml'

model=DiT_Llama_3B_patch2
batch_size=512
lr=1e-4
precision=bf16

exp_name=${model}_bs${batch_size}_lr${lr}_${precision}_qknorm

torchrun --nproc_per_node 8 train.py \
    --model ${model} \
    --data_path ${train_data_root} \
    --results_dir results/${exp_name} \
    --micro_batch_size 32 \
    --global_batch_size ${batch_size} --lr ${lr} \
    --data_parallel fsdp \
    --max_steps 3000000 \
    --ckpt_every 10000 --log_every 100 \
    --precision ${precision} --grad_precision fp32 --qk_norm
