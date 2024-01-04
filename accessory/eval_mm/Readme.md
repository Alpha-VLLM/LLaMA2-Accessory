# Evaluation

## Dependencies

```bash
pip install pycocoevalcap tqdm icecream textdistance editdistance nltk scikit-learn
```

## Annotation
We provide the processed officials annotations and convert them into unified format in *annotations* file.

## File Structure
```
├── eval_mm/
│   ├── data/
│   │   ├── MME_Benchmark_release_version/
│   │   │   ├── artwork
│   │   │   ├── celebrity
│   │   │   ├── ...
│   │   ├── coco/
│   │   ├── gqa/
│   │   └── ...
│   ├── annotations/
│   ├── utils/
│   ├── scripts/
│   ├── evaluate.py
│   ├── inference_image_sphinx.py
│   ├── sphinx.py
│   └── infographicsvqa_eval.py
```
## Run scripts


```bash
# download model weight from https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/main/SPHINX
pretrained_path=[PATH_TO_PRETRAINED_MODEL]

# evaluate all benchmarks
srun python inference_image_sphinx.py \
--dataset all \
--pretrained_path ${pretrained_path} \
--batch_size 32 \
--max_seq_length 4096 \
--model_parallel_size 2


# only evaluate mme and okvqa benchmarks
srun python inference_image_sphinx.py \
--dataset mme okvqa \
--pretrained_path ${pretrained_path} \
--batch_size 32 \
--max_seq_length 4096 \
--model_parallel_size 2
```



## Download Data
Run the scripts under *eval_mm*
```bash
# COCO images

mkdir -p data/coco && cd data/coco

wget http://images.cocodataset.org/zips/train2014.zip && unzip train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip && unzip val2014.zip
wget http://images.cocodataset.org/zips/test2015.zip && unzip test2015.zip

cd ../..

# VQAV2 Annotations
mkdir -p data/vqav2 && cd data/vqav2
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip && unzip v2_Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip && unzip v2_Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip && unzip v2_Annotations_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip && unzip v2_Questions_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip && unzip v2_Questions_Test_mscoco.zip
cd ../..

# OKVQA Annotations
mkdir -p data/okvqa && cd data/okvqa
# download annotations and questions
wget https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip && unzip mscoco_train2014_annotations.json.zip
wget https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip && unzip OpenEnded_mscoco_train2014_questions.json.zip
wget https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip && unzip mscoco_val2014_annotations.json.zip
wget https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip && unzip OpenEnded_mscoco_val2014_questions.json.zip

cd ../..

# TextVQA Annotations
mkdir -p data/textvqa && cd data/textvqa

# download images
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip && unzip train_val_images.zip

# download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_train_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_train_questions.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_val_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_val_questions.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_val.jsonl

cd ../..

mkdir -p data/vizwiz && cd data/vizwiz
# download images
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip && unzip val.zip
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip && unzip test.zip

# download converted files
# val
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_val_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_val_questions.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_val.jsonl
# test
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_test.jsonl
cd ../..

mkdir -p data/docvqa && cd data/docvqa

# DocVQA
# download images and annotations from https://www.docvqa.org/datasets

cd ../..

# ChartQA
mkdir -p data/chartqa && cd data/chartqa

# download images from https://drive.google.com/file/d/1Lm_w6zeET1Hyl_9ks6w5nEsgpoyPHalV/view

cd ../..

# GQA 
mkdir -p data/gqa && cd data/gqa

# download images
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
unzip images.zip

cd ../..

# OCRVQA

mkdir -p data/ocrvqa && cd data/ocrvqa

# download images by following instructions at https://ocr-vqa.github.io/kvqa_ProjectFiles/README.txt

cd ../..

# AI2Diagram
mkdir -p data/ai2diagram && cd data/ai2diagram

# download images
wget https://ai2-public-datasets.s3.amazonaws.com/diagrams/ai2d-all.zip

cd ../..

# ScienceQA
mkdir -p data/scienceqa/images && cd data/scienceqa/images

# download images
wget https://scienceqa.s3.us-west-1.amazonaws.com/images/test.zip && unzip test.zip

cd ../../..

# MME

# download MME images https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/blob/Evaluation/README.md
# place the images under data/MME_Benchmark_release_version

# Seedbenchv1

mkdir -p data/SEED_Bench && cd data/SEED_Bench

wget https://huggingface.co/datasets/AILab-CVC/SEED-Bench/resolve/main/SEED-Bench-image.zip?download=true && unzip SEED-Bench-image.zip
cd ../..

# Seedbenchv2

mkdir -p data/seedbenchV2 && cd data/seedbenchV2

wget https://huggingface.co/datasets/AILab-CVC/SEED-Bench-2/resolve/main/cc3m-image.zip?download=true  && unzip cc3m-image.zip
# download SEED-Bench-2.zip.* from https://huggingface.co/datasets/AILab-CVC/SEED-Bench-2/tree/main
cat SEED-Bench-2.zip.* > SEED-Bench-2.zip
unzip SEED-Bench-2.zip

cd ../..

```
