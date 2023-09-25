DIR="data/MM-Vet/"

pretrained_type=consolidated

image_folder_path="$DIR"/images
answers_file_path="MMVet_result/"${model_name}"/"${model_name}".json"



# Modify the following parameters
model_name="your_model_name"

pretrained_path="path/to/pretrained"
llama_config="path/to/params.json"
tokenizer_path="path/to/tokenizer.model"


# mode settings:
    # inference: Get model answers.
    # eval: Use GPT4 to score the mod's answers against the GPT4 answers.
    # all: Inferring, outputting results for models.
CUDA_VISIBLE_DEVICES=3 torchrun --nproc-per-node=1 --master_port=23263 src/eval_mmvet.py \
  --model_name ${model_name} \
  --llama_type llama_qformerv2 \
  --llama_config ${llama_config} \
  --tokenizer_path ${tokenizer_path} \
  --pretrained_path ${pretrained_path} \
  --model_parallel_size 1 \
  --openai_key sk-xxxxxxxx \
  --image_folder ${image_folder_path} \
  --question_file "$DIR"/mm-vet.json \
  --answers_file ${answers_file_path} \
  --use_sub_set False \
  --mode all \
  2>&1 | tee model.log

