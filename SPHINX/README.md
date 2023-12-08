# 🔥 SPHINX: A Mixer of Tasks, Domains, and Embeddings

Official implementation of ['SPHINX: A Mixer of Tasks, Domains, and Embeddings Advances Multi-modal Large Language Models'](https://github.com/Alpha-VLLM/LLaMA2-Accessory/blob/main/SPHINX/SPHINX_paper.pdf).

Try out our [web demo 🚀](http://imagebind-llm.opengvlab.com/) here!
<p align="center">
   🤗 <a href="https://huggingface.co/Alpha-VLLM/SPHINX" target="_blank">HF Repo</a> • 👋 join our <a href="https://github.com/Alpha-VLLM/LLaMA2-Accessory/blob/main/docs/wechat.md" target="_blank">WeChat</a>
</p>

## News
* **[2023-11-17]** We release SPHINX-V2, featuring the same architecture but with enhanced and broader capabilities! 🔥🔥🔥
* **[2023-11-09]** We release the [technical report](https://github.com/Alpha-VLLM/LLaMA2-Accessory/blob/main/SPHINX/SPHINX_paper.pdf) of SPHINX 🔥.
* **[2023-10-17]** We release the demo, code, and model of SPHINX 🎉.

## Introduction

We present $\color{goldenrod}{SPHINX}$, a versatile multi-modal large language model (MLLM) with a mixer of training tasks, data domains, and visual embeddings. 

- **Task Mix.** For all-purpose capabilities, we mix a variety of vision-language tasks for mutual improvement: VQA, REC, REG, OCR, DET, POSE, REL DET, T2I, etc.

- **Embedding Mix.** We capture robust visual representations by fusing distinct visual architectures, pretraining, and granularity.

- **Domain Mix.** For data from real-world and synthetic domains, we mix the weights of two domain-specific models for complementarity.

<p align="center">                                                                                                                                          <img src="figs/pipeline.png"/ width="90%"> <br>
</p>

On top of SPHINX, we propose to further mix visual scales and sub-images for better capture fine-grained semantics on high-resolution images.
<p align="center">                                                                                                                                          
  <img src="figs/pipeline2.png"/ width="90%"> <br>
</p>

## Inference
### Installation
+ SPHINX is built upon LLaMA2-Accessory, please follow the instructions [here](https://llama2-accessory.readthedocs.io/en/latest/install.html) for environment setup.
+ **Important 🔦:** For flexible instantiation of SPHINX models, please set up the LLaMA2-Accessory repo to your python environment.
  ``` bash
  # go to the root directory of LLaMA2-Accessory
  cd LLaMA2-Accessory
  # install LLaMA2-Accessory 
  pip install -e .
  ```
  After this, you will be able to invoke `import accessory` or `import SPHINX` without the restriction of working directory.
+ To enable the segmentation ability shown in our official demo, SAM is also needed:
    ``` bash
    pip install git+https://github.com/facebookresearch/segment-anything.git
    ```

### Weights

We release the following checkpoints:

| Name         | Architecture                                      | Checkpoint                                                   |
| ------------ | ------------------------------------------------- | ------------------------------------------------------------ |
| SPHINX       | [llama_ens](../accessory/model/LLM/llama_ens.py)  | [Hugging face](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/tree/main/finetune/mm/SPHINX/SPHINX)/[Baidu](https://pan.baidu.com/s/1HE6NoF1ZawhMgJxeh9r2kQ?pwd=46s7)(提取码：46s7) |
| SPHINX-1K    | [llama_ens5](../accessory/model/LLM/llama_ens.py) | [Hugging face](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/tree/main/finetune/mm/SPHINX/SPHINX-1k)/[Baidu](https://pan.baidu.com/s/1SRfyFGJdapaUTgYZOAdXyg?pwd=pua9)(提取码：pua9) |
| SPHINX-v2-1k | [llama_ens5](../accessory/model/LLM/llama_ens.py) | [Hugging face](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/tree/main/finetune/mm/SPHINX/SPHINX-v2-1k)/[Baidu](https://pan.baidu.com/s/1PKCf515EGmSnSZ8teERHjQ?pwd=88z0)(提取码：88z0) |

*Note that SPHINX-1K was previously called Long-SPHINX*

Please download them to your own machine. The file structure should appear as follows:

```
path/to/checkpoint
├── consolidated.00-of-02.model.pth
├── consolidated.01-of-02.model.pth
├── tokenizer.model
├── config.json
└── meta.json
```

### Inference

#### Single-GPU Inference
```python
from SPHINX import SPHINXModel
from PIL import Image
import torch

# Besides loading the `consolidated.*.pth` model weights, from_pretrained will also try to 
# use `tokenizer.model', 'meta.json', and 'config.json' under `pretrained_path` to configure
# the `tokenizer_path`, `llama_type`, and `llama_config` of the model. You may also override
# the configurations by explicitly specifying the arguments
model = SPHINXModel.from_pretrained(pretrained_path="path/to/checkpoint", with_visual=True)

image = Image.open("examples/1.jpg")
qas = [["What's in the image?", None]]

with torch.cuda.amp.autocast(dtype=torch.float16):
    response = model.generate_reponse(qas, image, max_gen_len=1024, temperature=0.9, top_p=0.5, seed=0)

print(response)

# if you wanna continue
qas[-1][-1] = response
qas.append(["Then how does it look like?", None])
with torch.cuda.amp.autocast(dtype=torch.float16):
    response2 = model.generate_reponse(qas, image, max_gen_len=1024, temperature=0.9, top_p=0.5, seed=0)

print(response2)
```

#### Multi-GPU inference
```python
from SPHINX import SPHINXModel
from PIL import Image
import torch
import torch.distributed as dist
import multiprocessing as mp

def main(world_size, rank) -> None:
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size,
        init_method=f"tcp://127.0.0.1:23560",
    )
    torch.cuda.set_device(rank)
    
    # mp_group tells the model which ranks will work together
    # through model parallel to compose a complete model.
    # When mp_group is None, a single-rank process group will
    # be created and used, which means model parallel size = 1 (not enabled)
    model = SPHINXModel.from_pretrained(
        pretrained_path="path/to/checkpoint", with_visual=True,
        mp_group=dist.new_group(ranks=list(range(world_size)))
    ) 
    
    # it's important to make sure that ranks within the same 
    # model parallel group should always receive the same input simultaneously
    image = Image.open("examples/1.jpg")
    qas = [["What's in the image?", None]]

    with torch.cuda.amp.autocast(dtype=torch.float16):
        response = model.generate_reponse(qas, image, max_gen_len=1024, temperature=0.9, top_p=0.5, seed=0)


if __name__ == "__main__":
    N_GPU = 2
    if N_GPU == 1:
        main(world_size=1, rank=0)
    elif N_GPU == 2:
        # You can use whatever method, e.g. torchrun, slurm, etc. for distributed launch
        # Just be sure to initialize torch distributed (by invoking dist.init_process_group)
        # before creating the SPHINX model if model parallel size > 1 is used
        mp.set_start_method("spawn")
        for rank in range(N_GPU):
            process = mp.Process(target=main, args=(N_GPU, rank))
            process.start()
    else:
        raise ValueError("Currently only 1 or 2 is supported for MODEL_PARALLEL_SIZE")
```
If torchrun is preferred, an example is [inference.py](inference.py):
```bash
torchrun --master_port=1112 --nproc_per_node=2 inference.py
```


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
--pretrained_path /path/to/checkpoint/
```
Explanation of each argument:

+ `--n_gpus`: Number of gpus to use. Utilizing more GPUs will alleviate memory usage on each GPU through model parallelism. Currently, this argument should be set to either 1 or 2, as support for *consolidated ckpt num < gpu num* is not yet available.
+ `--tokenizer_path`: Path to the official LLaMA2 tokenizer. Note that the tokenizer file is the same for both LLaMA and LLaMA2. You may download it from [here](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/blob/main/config/tokenizer.model).
+ `--llama_type`: The model architecture of SPHINX is defined in [accessory/model/LLM/llama_ens.py](../accessory/model/LLM/llama_ens.py),  and specifying `--llama_type=llama_ens` tells the demo program to use this architecture.
+ `--pretrained_path`: The path to pretrained checkpoint.

#### SPHINX-1k & SPHINX-v2-1k
Execute the following command for demo hosting:
``` bash
cd LLaMA2-Accessory/accessory
python demos/multi_turn_mm_box.py --n_gpus=2 \
--tokenizer_path=/path/to/tokenizer.model --llama_type=llama_ens5 \
--pretrained_path /path/to/checkpoint/
```
Explanation:
+ `--llama_type`: The model architecture of SPHINX-1k is defined in [accessory/model/LLM/llama_ens5.py](../accessory/model/LLM/llama_ens5.py), and specifying `--llama_type=llama_ens5` tells the demo program to use this architecture.
