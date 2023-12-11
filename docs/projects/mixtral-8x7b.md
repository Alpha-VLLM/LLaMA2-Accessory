# mixtral-8x7b

[mixtral-8x7b](https://huggingface.co/someone13574/mixtral-8x7b-32kseqlen) is a Mixture-of-Expert (MoE) model. In this
tutorial, we will introduce how to inference with and to finetune the model.

:::{admonition} Online Demo of Finetuned Model 🚀🚀🚀
:class: tip

We host a web demo at <https://dfc02190724c71dd5b.gradio.live/>, which shows a mixtral-8x7b model finetuned on 
[evol-codealpaca-v1](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1) and 
[ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k), with LoRA and Bias tuning. 
Please note that this is a temporary link, and we will update our official permanent link today.
:::

## Features
With LLaMA2-Accessory, mixtral-8x7b enjoys the following features:
1. Distributed MoE (namely instantiating experts on multiple processes/gpus)
2. Load Balancing Loss
3. Tensor Parallel and FSDP for efficiently training
4. Distributed and/or quantized inference

## Model Implementation
LLaMA2-Accessory implements mixtral-8x7b in {link2repo}`[mistral.py](accessory/model/LLM/mistral.py)`; it also 
implements a PEFT version (supporting bias/norm/LoRA tuning) in {link2repo}`[mistral_peft.py](accessory/model/LLM/mistral_peft.py)`

## Install
Please follow the [instructions here](https://llama2-accessory.readthedocs.io/en/latest/install.html) to install
LLaMA2-Accessory, which is an easy-to-use and comprehensive toolkit for LLM development.

## Prepare Checkpoint
Given the official mixtral-8x7b checkpoints, a step of format conversion is needed to make them usable by
LLaMA2-Accessory. We have released the off-the-shelf converted checkpoints. Alternatively, you can convert them 
by yourself according to the following guides.
### A. Download Converted Checkpoints
The converted checkpoints are released at [HuggingFace](https://huggingface.co/Alpha-VLLM/MoE-Mixtral-7B-8Expert/tree/main/converted),
please download all files in the folder to your machine. 
### B. Convert by Yourself

#### 1. prepare the original checkpoints
The original checkpoints are available at https://huggingface.co/someone13574/mixtral-8x7b-32kseqlen, please first
download the 10 splits and then cat them into one follow the official guides. After this step, you should have the 
`consolidated.00.pth` file.

#### 2. convert

Downlaod the [split.py](https://huggingface.co/Alpha-VLLM/MoE-Mixtral-7B-8Expert/blob/main/converted/split.py) script and *put it in the same directory as `consolidated.00.pth`*. Run the following
command to conduct conversion:
```bash
python split.py
```
After running, you should see a folder named `converted` created, with eight `consolidated.**-of-08.model.pth` files
therein. 

#### 3. prepare other resources
Finally, please download the following three files from [our HuggingFace repo](https://huggingface.co/Alpha-VLLM/MoE-Mixtral-7B-8Expert/tree/main/converted):
:::{card}
[config.json](https://huggingface.co/Alpha-VLLM/MoE-Mixtral-7B-8Expert/blob/main/converted/config.json)
[meta.json](https://huggingface.co/Alpha-VLLM/MoE-Mixtral-7B-8Expert/blob/main/converted/meta.json)
[tokenizer.model](https://huggingface.co/Alpha-VLLM/MoE-Mixtral-7B-8Expert/blob/main/converted/tokenizer.model)
:::
and put them under the `converted` directory, next to the weight files you obtained in the previous step.

### Result
No matter you have downloaded or converted the checkpoints on your own, you should finally get the following file structure:
```
path/to/converted
# model weights
├── consolidated.00-of-04.model.pth
├── consolidated.01-of-04.model.pth
├── consolidated.02-of-04.model.pth
├── consolidated.03-of-04.model.pth
# spm-format tokenizer 
├── tokenizer.model
# model configuration
├── config.json
# meta information, currently only contains model type
└── meta.json
```


## Inference
### Simple Inference
You can run inference on 8, 4, 2, or 1 GPUs. With tensor parallel and distributed MoE, the more GPUs you use, the 
less memory and computation load exists on each individual GPU. The following code exemplifies the inference process.
```python
from accessory.model.meta import MetaModel

import random 
import numpy as np

import torch
import torch.distributed as dist
import multiprocessing as mp

def main(world_size, rank) -> None:
    # specify random seed to ensure consistent token sampling among model parallel ranks
    random.seed(0)
    torch.random.manual_seed(0)
    np.random.seed(0)
    
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size,
        init_method=f"tcp://127.0.0.1:23560",
    )
    torch.cuda.set_device(rank)
    
    # mp_group identifies which ranks will work collaboratively through model parallelism
    model = MetaModel.from_pretrained("/path/to/converted", max_seq_len=2048,
                                      mp_group=dist.new_group(ranks=list(range(dist.get_world_size()))))

    prompt = "The best programming language in the world is"

    response = model.generate([prompt], images=None, max_gen_len=512)[0]
    if rank == 0:  # without this filter, the response will be printed for `world_size` times
        print(response)
    # or if you want to generate the response token by token
    response = None
    for response_in_progress in model.stream_generate(prompt, image=None, max_gen_len=512):
        response = response_in_progress['text']
        if rank == 0:
            print(response)


if __name__ == "__main__":
    N_GPU = 8 # 1, 2, 4, or 8
    if N_GPU == 1:
        main(world_size=1, rank=0)
    elif N_GPU > 1:
        # You can use whatever method, e.g. torchrun, slurm, etc. for distributed launch
        # Just be sure to initialize torch distributed (by invoking dist.init_process_group)
        # before creating the model if model parallel size > 1 is used
        mp.set_start_method("spawn")
        for rank in range(N_GPU):
            process = mp.Process(target=main, args=(N_GPU, rank))
            process.start()
    else:
        raise ValueError
```

A thorough tutorial over the inference with LLaMA2-Accessory can be found in the 
[document](https://llama2-accessory-temp.readthedocs.io/en/latest/inference.html).

### Host Local Demo
LLaMA2-Accessory provides a series of gradio demos for efficient interaction with your model. To host a local demo
for the pretrained mixtral-8x7b model, follow the steps below:
```bash
cd LLaMA2-Accessory/accessory
torchrun --nproc-per-node=$N_GPUS_TO_USE --master-port=$PORT demos/single_turn.py \
--pretrained_path $PATH_TO_CONVERTED
```
As we have mentioned in the [Simple Inference](#simple-inference) section, `$N-GPUS-TO-USE` can be 1, 2, 4, or 8. 
`$PATH_TO_CONVERTED` should be the directory containing the converted checkpoints, and `$PORT` can be any free port.

:::{tip}

The `demos/single_turn.py` file was designed to support both pretrained models and models finetuned with alpaca-style template. 
For pretrained models, please set the `system_prompt` optional to `None` in the Web GUI. 
See the LLaMA2-Accessory [document](https://llama2-accessory.readthedocs.io/en/latest/) to know more about
[finetuning](https://llama2-accessory.readthedocs.io/en/latest/finetune/index.html) 
and [inference](https://llama2-accessory-temp.readthedocs.io/en/latest/inference.html).
:::

## Finetuning
LLaMA2-Accessory supports both full-parameter and parameter-efficient finetuning of mixtral-8x7b. It also 
supports the load balancing regularization loss. More advanced MoE support will come soon.

### Data
We use the following datasets to exemplify finetuning:
+ [evol-codealpaca-v1](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1)
+ [ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)

The two files are referred to by the [dialog_ultrachat200kWizardcode.yaml](https://github.com/Alpha-VLLM/LLaMA2-Accessory/accessory/configs/data/finetune/sg/dialog_ultrachat200kWizardcode.yaml) 
file, which is then used by the `*.sh` experiments shown below to define the data for fientuning. Note that the data need
to be processed to match the format usable by LLaMA2-Accessory. For convenience, we provide the processed data files for 
[💾evol-codealpaca-v1](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/data/evol-codealpaca-v1/wizardCode.json) and
[💾ultrachat_200k](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/data/ultrachat_200k_train_sft.json).
Please move them to the position specified by `dialog_ultrachat200kWizardcode.yaml`


### Full Finetune
```bash
cd LLaMA2-Accessory/accessory
srun -n32 --gres=gpu:8 --ntasks-per-node=8 bash \
exps/finetune/sg/dialog_ultrachat200kWizardcode_mistral.sh \
/path/to/converted/mixtral-8x7b-32kseqlen \
/path/to/converted/mixtral-8x7b-32kseqlen/config.json \
/path/to/converted/mixtral-8x7b-32kseqlen/tokenizer.model
```
### PEFT
```bash
cd LLaMA2-Accessory/accessory
srun -n16 --gres=gpu:8 --ntasks-per-node=8 bash \
exps/finetune/sg/dialog_ultrachat200kWizardcode_mistralPeft.sh \
/path/to/converted/mixtral-8x7b-32kseqlen \
/path/to/converted/mixtral-8x7b-32kseqlen/config.json \
/path/to/converted/mixtral-8x7b-32kseqlen/tokenizer.model
```

**Finetuned Model Release:**

+ [🤗checkpoint](https://huggingface.co/Alpha-VLLM/MoE-Mixtral-7B-8Expert/tree/main/finetuned/peft)

**Host Local Demo**
```bash
cd LLaMA2-Accessory/accessory
python demos/multi_turn.py --n_gpus $N_GPUS_TO_USE --pretrained_path $PATH_TO_FINETUNED
```

See the LLaMA2-Accessory [document](https://llama2-accessory.readthedocs.io/en/latest/) to know more about
[finetuning](https://llama2-accessory.readthedocs.io/en/latest/finetune/index.html) 
and [inference](https://llama2-accessory-temp.readthedocs.io/en/latest/inference.html).


## Acknowledgement
+ [@dzhulgakov](https://github.com/dzhulgakov) for [llama-mistral](https://github.com/dzhulgakov/llama-mistral)
+ [@mistralai](https://github.com/mistralai) for [megablocks](https://github.com/mistralai/megablocks-public)