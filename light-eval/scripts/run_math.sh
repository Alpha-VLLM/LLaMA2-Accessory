task=math
pretrained_type=meta_ori
pretrained_path=/path/to/your/model_dir
llama_config=/path/to/your/config
tokenizer_path=/path/to/your/tokenizer
data_dir='data/math'

nproc_per_node=1
# model_parallel=1
master_port=23456

exp_name=your/model/name
mkdir -p logs/"$exp_name"

torchrun --nproc-per-node="$nproc_per_node" --master_port "$master_port" src/eval_"$task".py \
    --pretrained_type "$pretrained_type" \
    --llama_config "$llama_config" \
    --tokenizer_path "$tokenizer_path" \
    --pretrained_path "$pretrained_path" \
    2>&1 | tee logs/"$exp_name"/"$task".log