# Multimodel evaluation
## environment settings
Before running the Light-eval, users must ensure that they have correctly installed and configured all necessary environments according to the instructions in the [Installation Document](../install.md).

## LLaVA-benchmark
### Prerequisites
**dataset**
```
├── data
│   └── LLaVA-benchmark
│       ├── images
│       │   ├── 001.jpg
│       │   ├── 002.jpg
│       │   └── ...
│       ├── answers_gpt4.jsonl
│       ├── context.jsonl
│       └── ...
└── ...
```
Store the images folder according to the file structure given above in `data`. 

The dataset is availabel at [🤗Hugging Face/liuhaotian](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild)



### evaluating


Please make sure the dataset is stored according to the storage structure described above. 

Change the parameters in `scripts/run_llavabenchmark.sh`:
`model_name`, `pretrained_path`, `llama_config`, `tokenizer_path`, `openai_key` and `mode` .



**mode settings:**

* **inference**: Get model answers.
* **eval**: Use GPT4 to score the modle's answers against the GPT4 answers.
* **show**: Output of the scored results
* **all**: Inferring, scoring, and outputting results for models.



After changing parameters, you can use following script to run LLaVA-benchmark evaluation code for your model.

**script**

```bash
sh scripts/run_llavabenchmark.sh
```



## MM-Vet benchmark
### Prerequisites
**dataset**
```
├── data
│   └── MM-Vet
│       ├── images
│       │   ├── v1_0.png
│       │   ├── v1_2.png
│       │   └── ...
│       ├── mm-vet.json
│       └── bard_set.json
└── ...
```
Store the images folder according to the file structure given above in `data` 

Download MM-Vet data [yuweihao/mm-vet.zip](https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip) and unzip the dataset file according to the format described above.


### evaluating

Please make sure the dataset is stored according to the storage structure described above. 

Change the parameters in `scripts/run_mmvet.sh`: 
`model_name`, `pretrained_path`, `llama_config`, `tokenizer_path`, `openai_key`, `use_sub_set` and `mode` .



**mode settings:**

* **inference**: Get model answers.
* **eval**: Use GPT4 to score the modle's answers against the GPT4 answers.
* **all**: Inferring, outputting results for models.



**use_sub_set:**

* **True**: use subset for evaluation.
* **False**: use the full dataset for evaluation.



After changing parameters, you can use following script to run MM_Vet benchmark evaluation code for your model.

**script**

```bash
sh scripts/run_mmvet.sh
```

