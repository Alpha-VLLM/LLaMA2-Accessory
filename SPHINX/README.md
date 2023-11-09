# 🔥 SPHINX: A Mixer of Tasks, Domains, and Embeddings

Official implementation of ['SPHINX: A Mixer of Tasks, Domains, and Embeddings Advances Multi-modal Large Language Models'](https://github.com/Alpha-VLLM/LLaMA2-Accessory/blob/main/SPHINX/SPHINX_paper.pdf).

Try out our [web demo 🚀](http://imagebind-llm.opengvlab.com/) here!
<p align="center">
   🤗 <a href="https://huggingface.co/Alpha-VLLM/SPHINX" target="_blank">HF Repo</a> • 👋 join our <a href="https://github.com/Alpha-VLLM/LLaMA2-Accessory/blob/main/docs/wechat.md" target="_blank">WeChat</a>
</p>

## News
* **[2023-11-09]** We release the [technical report](https://github.com/Alpha-VLLM/LLaMA2-Accessory/blob/main/SPHINX/SPHINX_paper.pdf) of SPHINX 🔥.
* **[2023-10-17]** We release the demo, code, and model of SPHINX 🎉.

## Introduction

We present $\color{goldenrod}{SPHINX}$, a versatile multi-modal large language model (MLLM) with a mixer of training tasks, data domains, and visual embeddings. 

- **Task Mix.** For all-purpose capabilities, we mix a variety of vision-language tasks for mutual improvement: VQA, REC, REG, OCR, etc.

- **Embedding Mix.** We capture robust visual representations by fusing distinct visual architectures, pre-training, and granularity.

- **Domain Mix.** For data from real-world and synthetic domains, we mix the weights of two domain-specific models for complementarity.

<p align="center">                                                                                                                                          <img src="figs/pipeline.png"/ width="90%"> <br>
</p>


## Inference
### Installation
+ SPHINX is built upon LLaMA2-Accessory, please follow the instructions [here](https://llama2-accessory.readthedocs.io/en/latest/install.html) for environment setup.
``` bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Weights
The checkpoints for SPHINX and Long-SPHINX (a multi-view augmented version) are released on [HuggingFace🤗](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/tree/main/finetune/mm/SPHINX). Please download them to your own machine. The file structure should appear as follows:
```
path/to/sphinx
├── consolidated.00-of-02.model.pth
└── consolidated.01-of-02.model.pth
```


### Simple Inference
We provide a simple script [inference.py](inference.py) to illustrate how to use SPHINX for inference:
```bash
python inference.py
```
Please modify the configuration variables within the script before running it.

### Host Local Demo
For thoes who want to host a demo like [our official one](http://imagebind-llm.opengvlab.com/) locally, this section provides a step-by-step guide. 
+ [SAM](https://github.com/facebookresearch/segment-anything.git) should be installed to enable segmentation. 
+ *If you're already familiar with the LLAMA2-Accessory toolkit, note that hosting a SPHINX demo follows the same pipeline as hosting demos for the other models supported by LLAMA2-Accessory.*


#### SPHINX
Execute the following command for demo hosting:
``` bash
cd LLaMA2-Accessory/accessory
python demos/multi_turn_mm_box.py --n_gpus=2 \
--tokenizer_path=/path/to/tokenizer.model --llama_type=llama_ens \
--pretrained_path /path/to/sphinx/
```
Explanation of each argument:

+ `--n_gpus`: Number of gpus to use. Utilizing more GPUs will alleviate memory usage on each GPU through model parallelism. Currently, this argument should be set to either 1 or 2, as support for *consolidated ckpt num < gpu num* is not yet available.
+ `--tokenizer_path`: Path to the official LLaMA2 tokenizer. Note that the tokenizer file is the same for both LLaMA and LLaMA2. You may download it from [here](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/blob/main/config/tokenizer.model).
+ `--llama_type`: The model architecture of SPHINX is defined in [accessory/model/LLM/llama_ens.py](../accessory/model/LLM/llama_ens.py),  and specifying `--llama_type=llama_ens` tells the demo program to use this architecture.
+ `--pretrained_path`: The path to pre-trained checkpoint.

#### Long-SPHINX
Execute the following command for demo hosting:
``` bash
cd LLaMA2-Accessory/accessory
python demos/multi_turn_mm_box.py --n_gpus=2 \
--tokenizer_path=/path/to/tokenizer.model --llama_type=llama_ens5 \
--pretrained_path /path/to/long-sphinx/
```
Explanation:
+ `--llama_type`: The model architecture of Long-SPHINX is defined in [accessory/model/LLM/llama_ens5.py](../accessory/model/LLM/llama_ens5.py), and specifying `--llama_type=llama_ens5` tells the demo program to use this architecture.
