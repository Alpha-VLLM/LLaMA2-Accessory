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

- **Embedding Mix.** We capture robust visual representations by fusing distinct visual architectures, pre-training, and granularity.

- **Domain Mix.** For data from real-world and synthetic domains, we mix the weights of two domain-specific models for complementarity.

<p align="center">                                                                                                                                          <img src="figs/pipeline.png"/ width="90%"> <br>
</p>

On top of SPHINX, we propose to further mix visual scales and sub-images for better capture fine-grained semantics on high-resolution images.
<p align="center">                                                                                                                                          
  <img src="figs/pipeline2.png"/ width="90%"> <br>
</p>

## Evaluation
We provide a comprehensive evaluation of $\color{goldenrod}{SPHINX}$ and showcase results across multiple benchmarks. 

### MLLM Benchmarks
| Method               | POPE | MME<sup>P</sup> | MME<sup>C</sup> | MMB  | MMB<sup>CN</sup> | SEED | LLava<sup>W</sup> | MM-Vet | CCbench | MathVista |
|----------------------|------|------------------|------------------|------|-------------------|------|---------------------|--------|---------|-----------|
| BLIP-2               | 85.3 | 1293.8           | -                | -    | -                 | 46.4 | 38.1                | 22.4   | -       | -         |
| InstructBLIP-7B      | -    | -                | -                | 36   | 23.7              | 53.4 | 60.9                | 26.2   | 12.1    | 25.3      |
| InstructBLIP-13B     | 78.9 | 1212.8           | -                | -    | -                 | -    | 58.2                | 25.6   | -       | -         |
| Shikra               | -    | -                | -                | 58.8 | -                 | -    | -                   | -      | -       | -         |
| LLaMA-AdapterV2      | -    | 1328.40          | 356.43           | -    | -                 | -    | -                   | -      | -       | -         |
| Qwen-VL-7B           | -    | -                | -                | 38.2 | 7.4               | 56.3 | -                   | -      | 5.5     | -         |
| Qwen-VL-7B-Chat      | -    | 1487.58          | **360.71**        | 60.6 | 56.7              | 58.2 | -                   | -      | **39.3** | -      |
| LLaVA1.5-7B          | 85.9 | -                | -                | 64.3 | 58.3              | 58.6 | 63.4                | 30.5   | 16.4    | -         |
| LLaVA1.5-13B         | 85.9 | -                | -                | **67.7** | **63.6** | 61.6 | 70.7                | 35.4   | 26.5    | -         |
| SPHINX               | 80.7 | 1476.1           | 322.2            | 66.9 | 56.2              | 69.14| 73.5                | 36.0   | 25.6    | 27.0      |
| Long-SPHINX          | **90.8** | **1560.2**    | 310.0            | 67.1 | 59.5              | **71.62** | **74.3** | **36.6** | 27.9 | **27.5** |


### Academic Task-oriented Benchmarks
| Method        | OKVQA | VQAV2 | VizWiz | GQA | VSR | ScienceQA | IconVQA | TextVQA | OCR-VQA |
|---------------|-------|-------|--------|-----|-----|-----------|---------|---------|---------|
| BLIP-2        | 45.9  | -     | 19.6   | 41.0| 50.9| -         | 40.6    | -       | 40.6    |
| InstructBLIP  | -     | -     | 33.4   | 49.5| 52.1| -         | 44.8    | -       | 44.8    |
| Shikra        | 47.2  | 77.4  | -      | -   | -   | -         | -       | -       | -       |
| MiniGPT-v2    | 57.8  | -     | **53.6** | 60.1| 62.9| -       | 51.5    | -       | -       |
| Qwen-VL-7B    | 58.6  | 79.5  | 35.2   | 59.3| 63.8| 67.1      | -       | **63.8**| **75.7**|
| Qwen-VL-7B-Chat| 56.6  | 78.2  | 38.9   | 57.5| 61.5| 68.2      | -       | 61.5    | 70.5    |
| LLaVA1.5-7B   | -     | 78.5  | 50     | 62  | -   | 66.8      | -       | 58.2    | -       |
| LLaVA1.5-13B  | -     | 80    | **53.6** | **63.3** | - | **71.6**| -       | 61.3    | -       |
| SPHINX        | 62.08 | 78.08 | 39.91  | 62.59 | 58.5| 66.01    | 50.35   | 51.63   | 66.01   |
| Long-SPHINX   | **62.21** | **80.2** | 46.75 | 62.88 | **65.42** | 70.01| **52.68** | 58.78 | 70.01   |


### REC benchmarks
| Methods           | RefCOCO (val) | RefCOCO (test-A) | RefCOCO (test-B) | RefCOCO+ (val) | RefCOCO+ (test-A) | RefCOCO+ (test-B) | RefCOCOg (val-u) | RefCOCOg (test-u) | Avg     |
|-------------------|---------------|------------------|------------------|----------------|-------------------|-------------------|------------------|-------------------|---------|
| UNINEXT           | 92.64         | 94.33            | 91.46            | 85.24          | 89.63             | 79.79             | 88.73            | 89.37             | 88.90   |
| G-DINO-L          | 90.56         | 93.19            | 88.24            | 82.75          | 88.95             | 75.92             | 86.13            | 87.02             | 86.60   |
| VisionLLM-H       | -             | 86.70            | -                | -              | -                 | -                 | -                | -                 | -       |
| OFA-L             | 79.96         | 83.67            | 76.39            | 68.29          | 76.00             | 61.75             | 67.57            | 67.58             | 72.65   |
| Shikra 7B         | 87.01         | 90.61            | 80.24            | 81.60          | 87.36             | 72.12             | 82.27            | 82.19             | 82.93   |
| Shikra 13B        | 87.83         | 91.11            | 81.81            | 82.89          | 87.79             | 74.41             | 82.64            | 83.16             | 83.96   |
| MiniGPT-v2 7B     | 88.69         | 91.65            | 85.33            | 79.97          | 85.12             | 74.45             | 84.44            | 84.66             | 84.29   |
| MiniGPT-v2 7B-chat| 88.06         | 91.29            | 84.30            | 79.58          | 85.52             | 73.32             | 84.19            | 84.31             | 83.70   |
| Qwen-VL-7B        | 89.36         | 92.26            | 85.34            | 83.12          | 88.25             | 77.21             | 85.58            | 85.48             | 86.45   |
| Qwen-VL-7B-Chat   | 88.55         | 92.27            | 84.51            | 82.82          | 88.59             | 76.79             | 85.96            | 86.32             | 85.74   |
| Sphinix           | 89.15         | 91.37            | 85.13            | 82.77          | 87.29             | 76.85             | 84.87            | 83.65             | 84.12   |
| Sphinix-Long      | **91.05**     | **92.65**        | **86.56**        | **86.64**      | **91.08**         | **80.35**         | **88.19**        | **88.35**         | **88.14**|



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
| SPHINX       | [llama_ens](../accessory/model/LLM/llama_ens.py)  | [here](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/tree/main/finetune/mm/SPHINX/SPHINX) |
| SPHINX-1K    | [llama_ens5](../accessory/model/LLM/llama_ens.py) | [here](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/tree/main/finetune/mm/SPHINX/SPHINX-1k) |
| SPHINX-v2-1k | [llama_ens5](../accessory/model/LLM/llama_ens.py) | [here](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/tree/main/finetune/mm/SPHINX/SPHINX-v2-1k) |

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
# use `tokenizer.json', 'meta.json', and 'config.json' under `pretrained_path` to configure
# the `tokenizer_path`, `llama_type`, and `llama_config` of the model. You may also override
# the configurations by explitly specifying the arguments
model = SPHINXModel.from_pretrained(pretrined_path="path/to/checkpoint", with_visual=True)

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
        pretrined_path="path/to/checkpoint", with_visual=True,
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
+ `--pretrained_path`: The path to pre-trained checkpoint.

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
