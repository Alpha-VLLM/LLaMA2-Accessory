DIR="data/LLaVA-benchmark"

image_folder_path="$DIR"/images

answers_file_path="LLaVA_benchmark/"$model_name"/answers.jsonl"
output_file_path="LLaVA_benchmark/"$model_name"/scores.jsonl"



# Modify the following parameters
model_name="your_model_name"

pretrained_path="path/to/pretrained"
llama_config="path/to/params.json"
tokenizer_path="path/to/tokenizer.model"


# mode settings:
    # inference: Get model answers.
    # eval: Use GPT4 to score the mod's answers against the GPT4 answers.
    # show: Output of the scored results
    # all: Inferring, scoring, and outputting results for models.
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master_port=23363 src/eval_llavabenchmark.py \
  --model_name ${model_name} \
  --llama_type llama_qformerv2 \
  --llama_config ${llama_config} \
  --tokenizer_path ${tokenizer_path} \
  --pretrained_path ${pretrained_path} \
  --model_parallel_size 1 \
  --image_folder ${image_folder_path} \
  --question_file "$DIR"/questions.jsonl \
  --answers_file ${answers_file_path} \
  --openai_key sk-xxxxxxxxxxxxx \
  --context "$DIR"/context.jsonl \
  --answer-list \
  "$DIR"/answers_gpt4.jsonl \
  ${answers_file_path} \
  --rule "$DIR"/rule.json \
  --output ${output_file_path} \
  --mode all \
  2>&1 | tee -a model.log
