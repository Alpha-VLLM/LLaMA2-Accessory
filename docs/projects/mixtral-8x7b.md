# mixtral-8x7b

[mixtral-8x7b](https://huggingface.co/someone13574/mixtral-8x7b-32kseqlen) is a Mixture-of-Expert (MoE) model. In this
tutorial, we will introduce how to inference with and to finetune the model.

:::{admonition} Online Demo of Finetuned Model 🚀🚀🚀
:class: tip

We host a web demo [💻here](http://106.14.127.192/), which shows a mixtral-8x7b model finetuned on 
[evol-codealpaca-v1](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1) and 
[ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k), with LoRA and Bias tuning.
:::

## Features
With LLaMA2-Accessory, mixtral-8x7b enjoys the following features:
1. [Two Implementations](#model-implementation)
2. Load Balancing Loss
3. Tensor Parallel and FSDP for efficiently training
4. Distributed and/or quantized inference
5. Multi-modal support

## Model Implementation
There are generally two approaches to implement the Mixture of Experts (MoE) layers:
1. The <span style="color: #00e0e0">base </span> implementation (Distribute Different Experts to Different GPUs): For example, given 8 experts and 4 GPUs, each GPU will be allocated
with two experts. This is the approach adopted by [DiscoResearch](https://huggingface.co/DiscoResearch/mixtral-7b-8expert)
and [llama-mistral](https://github.com/dzhulgakov/llama-mistral).
2. The <span style="color: #00e0e0">sparse </span> implementation (Distribute a Part of Each Expert to Every GPU): For example, given 8 experts and 4 GPUs, each GPUs will hold 1/4 of
each expert. Such portioning of individual expert is achieved by splitting along the FFN hidden dim. This is the approach
officially adopted by [MistralAI](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1). We call it the *sparse* approach
because it [reformulates MoE computation to block-sparse operations](https://arxiv.org/pdf/2211.15841.pdf).

LLaMA2-Accessory supports **both** implementations. The two implementations are completely interchangeable. However, 
Benefited from the meticulously designed operators and the desirable nature of balanced computation load among GPUs, 
the second implementation is generally more efficient. On the other hand, the first one may be easier for beginners
to understand, and is also easier to be combined with LoRA.

The <span style="color: #00e0e0">base</span> implementation of Mixtral-8x7b is in {link2repo}`[mixtral.py](accessory/model/LLM/mixtral.py)`;
a corresponding PEFT version (supporting bias/norm/LoRA tuning) is in {link2repo}`[mixtral_peft.py](accessory/model/LLM/mixtral_peft.py)`.
For the <span style="color: #00e0e0">base</span> implementation, we prioritize simplicity over efficiency.


The <span style="color: #00e0e0">sparse</span> implementation of Mixtral-8x7b is in {link2repo}`[mixtral_sparse.py](accessory/model/LLM/mixtral_sparse.py)`.
Based on the implementation, {link2repo}`[mixtral_sparse_ens.py](accessory/model/LLM/mixtral_sparse_ens.py)` implements a
multi-modal model, with similar architecture to [SPHINX](https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/main/SPHINX)
but using mixtral-8x7b instead of LLaMA2 as LLM backbone. We are actively working on this multi-modal model and 
the checkpoint will be released soon. For the sparse implementation, we place greater emphasis on 
efficiency. Specifically, we have referred to the official implementation and introduced some efficient
operators from [megablocks](https://github.com/stanford-futuredata/megablocks/)
and [stk](https://github.com/stanford-futuredata/stk).


## Install
Please follow the [instructions here](https://llama2-accessory.readthedocs.io/en/latest/install.html) to install
LLaMA2-Accessory, which is an easy-to-use and comprehensive toolkit for LLM development. If you want to use
the <span style="color: #00e0e0">sparse</span> implementation of mixtral-8x7b, 
please also install [megablocks](https://github.com/stanford-futuredata/megablocks/)
and [stk](https://github.com/stanford-futuredata/stk) according their the official guides.

## Prepare Checkpoint
Given the official mixtral-8x7b checkpoints, a step of format conversion is needed to make them usable by
LLaMA2-Accessory. We have released the off-the-shelf converted checkpoints. Alternatively, you can convert them 
by yourself according to the following guides.

:::{important}

Despite being two equivalent implementations of the same model, the checkpoints of the 
<span style="color: #00e0e0">base</span> and the <span style="color: #00e0e0">sparse</span>
implementations are not interchangeable. Please ensure to use the correct checkpoint.
:::

### A. Download Converted Checkpoints
The converted checkpoints are released at 🤗HuggingFace. For the <span style="color: #00e0e0">base</span> implementation,
the checkpoint is provided at [🤗base checkpoint](https://huggingface.co/Alpha-VLLM/MoE-Mixtral-7B-8Expert/tree/main/converted); For
the <span style="color: #00e0e0">sparse</span> implementation, the checkpoint is provided 
at [🤗sparse checkpoint](https://huggingface.co/Alpha-VLLM/MoE-Mixtral-7B-8Expert/tree/main/converted_sparse).
please download all the files in the folders to your machine. 
### B. Convert by Yourself

#### 1. prepare the original checkpoints
The original checkpoints (torrent release) are available at https://huggingface.co/someone13574/mixtral-8x7b-32kseqlen, 
please first download the 10 splits and then cat them into one follow the official guides. After this step, you should 
have the `consolidated.00.pth` file.

#### 2. convert

::::{grid} 2
:::{grid-item-card}

For <span style="color: #00e0e0">base</span> implementation
^^^
Download the [split.py](https://huggingface.co/Alpha-VLLM/MoE-Mixtral-7B-8Expert/blob/main/converted/split.py) script 
and *put it in the same directory as `consolidated.00.pth`*. Run the following command to convert:
```bash
python split.py
```
After running, you should see a folder named `converted` created, with eight `consolidated.**-of-08.model.pth` files
therein. 
:::

:::{grid-item-card}

For <span style="color: #00e0e0">sparse</span> implementation
^^^
Download the [split_sparse.py](https://huggingface.co/Alpha-VLLM/MoE-Mixtral-7B-8Expert/blob/main/converted_sparse/split_sparse.py)
script and *put it in the same directory as `consolidated.00.pth`*. Run the following command to convert:
```bash
python split_sparse.py
```
After running, you should see a folder named `converted_sparse` created, with eight `consolidated.**-of-08.model.pth` 
files therein.
:::
::::


#### 3. prepare other resources

::::{grid} 2
:::{grid-item-card}

For <span style="color: #00e0e0">base</span> implementation
^^^
Finally, please download the following three files from [our HuggingFace repo](https://huggingface.co/Alpha-VLLM/MoE-Mixtral-7B-8Expert/tree/main/converted):

+ [config.json](https://huggingface.co/Alpha-VLLM/MoE-Mixtral-7B-8Expert/blob/main/converted/config.json)
+ [meta.json](https://huggingface.co/Alpha-VLLM/MoE-Mixtral-7B-8Expert/blob/main/converted/meta.json)
+ [tokenizer.model](https://huggingface.co/Alpha-VLLM/MoE-Mixtral-7B-8Expert/blob/main/converted/tokenizer.model)

and put them under the `converted` directory, next to the weight files you obtained in the previous step.
:::
:::{grid-item-card}

For <span style="color: #00e0e0">sparse</span> implementation
^^^
Finally, please download the following three files from [our HuggingFace repo](https://huggingface.co/Alpha-VLLM/MoE-Mixtral-7B-8Expert/tree/main/converted):

+ [config.json](https://huggingface.co/Alpha-VLLM/MoE-Mixtral-7B-8Expert/blob/main/converted_sparse/config.json)
+ [meta.json](https://huggingface.co/Alpha-VLLM/MoE-Mixtral-7B-8Expert/blob/main/converted_sparse/meta.json)
+ [tokenizer.model](https://huggingface.co/Alpha-VLLM/MoE-Mixtral-7B-8Expert/blob/main/converted_sparse/tokenizer.model)

and put them under the `converted_sparse` directory, next to the weight files you obtained in the previous step.
:::
::::

### Result
No matter you have downloaded or converted the checkpoints on your own, you should finally get the following file structure:
```
path/to/converted OR path/to/converted_sparse
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
    
    pretrained_path = "/path/to/converted"  # converted checkpoints of either base or sparse format
    # mp_group identifies which ranks will work collaboratively through model parallelism
    model = MetaModel.from_pretrained(pretrained_path, max_seq_len=2048,
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
[document](https://llama2-accessory-temp.readthedocs.io/en/latest/inference.html). In the above example, 
`pretrained_path` should be replaced with the real path of the checkpoints prepared in the previous section. 
The `from_pretrained` method will then probe the `meta.json` file in the given path to discern the type of 
llm used, namely the `llama_type` argument for initializing a Meta model. For the 
<span style="color: #00e0e0">base</span> implementation, `llama_type` is `mixtral`; otherwise for the
<span style="color: #00e0e0">sparse</span> implementation, `llama_type` is `mixtral_sparse`.


### Host Local Demo
LLaMA2-Accessory provides a series of gradio demos for efficient interaction with your model. To host a local demo
for the pretrained mixtral-8x7b model, follow the steps below:
```bash
cd LLaMA2-Accessory/accessory
torchrun --nproc-per-node=$N_GPUS_TO_USE --master-port=$PORT demos/single_turn.py \
--pretrained_path $PRETRAINED_PATH
```
As we have mentioned in the [Simple Inference](#simple-inference) section, `$N-GPUS-TO-USE` can be 1, 2, 4, or 8. 
`$PRETRAINED` should be the directory containing the converted 
(<span style="color: #00e0e0">base</span> or <span style="color: #00e0e0">sparse</span>) checkpoints,
and `$PORT` can be any free port.

:::{tip}

`demos/single_turn.py` file was designed to support both pretrained models and models finetuned with alpaca-style template. 
For pretrained models, please set the `system_prompt` option to `None` in the Web GUI. 
See the LLaMA2-Accessory [document](https://llama2-accessory.readthedocs.io/en/latest/) to know more about
[finetune](https://llama2-accessory.readthedocs.io/en/latest/finetune/index.html) 
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
[💾evol-codealpaca-v1](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/blob/main/data/evol-codealpaca-v1/wizardCode.json) and
[💾ultrachat_200k](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/blob/main/data/ultrachat_200k_train_sft.json).
Please move them to the position specified by `dialog_ultrachat200kWizardcode.yaml`


### Full Finetune
For the <span style="color: #00e0e0">base</span> implementation:
```bash
cd LLaMA2-Accessory/accessory
srun -n32 --gres=gpu:8 --ntasks-per-node=8 bash \
exps/finetune/sg/dialog_ultrachat200kWizardcode_mixtral.sh \
/path/to/converted \
/path/to/converted/config.json \
/path/to/converted/tokenizer.model
```
For the <span style="color: #00e0e0">sparse</span> implementation, change `dialog_ultrachat200kWizardcode_mixtral.sh`
to `dialog_ultrachat200kWizardcode_mixtralSparse.sh` (where the only different is changing the `llama_type` argument
from `mixtral` to `mixtral_sparse`), and `/path/to/converted` to `path/to/converted_sparse`.
### PEFT
```bash
cd LLaMA2-Accessory/accessory
srun -n16 --gres=gpu:8 --ntasks-per-node=8 bash \
exps/finetune/sg/dialog_ultrachat200kWizardcode_mixtralPeft.sh \
/path/to/converted \
/path/to/converted/config.json \
/path/to/converted/tokenizer.model
```

**Finetuned Model Release:**

+ [🤗checkpoint](https://huggingface.co/Alpha-VLLM/MoE-Mixtral-7B-8Expert/tree/main/finetuned/sg)

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
