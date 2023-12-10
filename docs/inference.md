# Inference
## Preparation
Before running the inference code, users must ensure that they have correctly installed and configured all necessary environments according to the instructions in the [Installation Document](./install.md).

For those who prefer not to delve into the extensive technical details, you can just execute `bash demos/start.sh` and enjoy.

### Prepare Checkpoints
**Our checkpoints are released at [🤗Hugging Face](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory)**

For newer versions of LLaMA2-Accessory, the meta/config/tokenizer information is saved together with the model weights,
so the saved checkpoints should present the following organization:
```
path/to/checkpoint
# model weights
├── consolidated.00-of-02.model.pth
├── consolidated.01-of-02.model.pth
# spm-format tokenizer 
├── tokenizer.model
# huggingface-format tokenizer
├── tokenizer.json
├── tokenizer_config.json
# model configuration
├── config.json
# meta information, currently only contains model type
└── meta.json
```
The Model weights are split and saved into `m` `consolidated.n-of-m.model.pth` files, where `m` is model parallel size.
Note that for **tokenizers**, both spm and huggingface formats are supported. Either of them is enough and there is 
*no* need to have files of both formats simultaneously.

:::{admonition} Legacy Checkpoints
:class: warning

Checkpoints saved by legacy versions of LLaMA2-Accessory only contain the `consolidated.*.model.pth` weight files. Such
checkpoints are still usable, but the information about `llama_type`, `llama_config` and `tokenizer_path` need to be
manually specified.
:::



## General Pipeline

### Model Instantiation

The static method `MetaModel.from_pretrained` support convenient instantiation of LLaMA2-Accessory models
based on pretrained checkpoints.

:::{card}
```{autodoc2-object} accessory.model.meta.MetaModel.from_pretrained
render_plugin = "myst"
no_index = true
```
:::

For checkpoints saved with newer versions of LLaMA2-Accessory, recourses like tokenizer and config are saved together
with model weights. In such cases, only `pretrained_path` need to be specified.
```python
from accessory.model.meta import MetaModel
model = MetaModel.from_pretrained("/path/to/pretrained")
```
Otherwise, **for legacy versions of checkpoints**, only `consolidated.*.pth` model weights are saved. In such cases,
explicit specification of `llama_type`, `llama_config` and `tokenizer_path` is needed. *Generally, this can be achieved
by assigning the three arguments with the same assignment used for training*.
```python
from accessory.model.meta import MetaModel
example_llama_type='llama_peft'
example_llama_config=[
    '/path/to/llama/7B/params.json',
    'configs/model/finetune/sg/llamaPeft_normBiasLora.json']
example_tokenizer_path='/path/to/tokenizer.model'
model = MetaModel.from_pretrained(
    "/path/to/pretrained", llama_type=example_llama_type,
    llama_config=example_llama_config, tokenizer_path=example_tokenizer_path
)
```
:::{tip}
See FAQ to know more about [llama_config](./faq.md#how-to-set-llama_config) and [tokenizer_path](./faq.md#how-to-set-tokenizer_path).
:::


### Input Construction & Response Generation

#### Pretrained Models
For pretrained models, namely those trained on large scale corpus without specific template, you can use any text as a
prompt to make the model continue writing the content.

```python
from accessory.model.meta import MetaModel

model = MetaModel.from_pretrained("/path/to/pretrained", max_seq_len=2048)

# for pretrained model (i.e. trained on large corpus without specific template)
prompt = "The best programming language in the world is"
response = model.generate([prompt], images=None, max_gen_len=512)[0]
print(response)
# or if you want to generate the response token by token
response = None
for response_in_progress in model.stream_generate(prompt, image=None, max_gen_len=512):
    response = response_in_progress['text']
print(response)
```

#### Single-turn-finetuned Models
After instruction finetuning, it is important to keep the template consistent across finetuning and inference. The
following shows an example of the Alpaca template, which is the default choice of LLaMA2-Accessory for single-turn
finetuning. If you have used different templates during finetuning, don't forget to continue using them for inference.
```python
from accessory.model.meta import MetaModel
from accessory.data.system_prompt import format_prompt

model = MetaModel.from_pretrained("/path/to/pretrained", max_seq_len=2048)

# for single-turn-finetuned model
instruction = "What's the best programming language in the world?"
prompt = format_prompt({"instruction": instruction}, sys_name="alpaca")
# prompt is equal to:
#   "Below is an instruction that describes a task."
#   "Write a response that appropriately completes the request.\n\n"
#   "### Instruction:\nWhat's the best programming language in the world?\n\n### Response:"

response = model.generate([prompt], images=None, max_gen_len=512)[0]
print(response)
# or if you want to generate the response token by token
response = None
for response_in_progress in model.stream_generate(prompt, image=None, max_gen_len=512):
    response = response_in_progress['text']
print(response)
```

#### Multi-turn-finetuned Models
Similar to the single-turn case, the template for multi-turn conversation should also be consistent across
finetuning and inference. The following shows an example of the default template used by LLaMA2-Accessory.
```python
from accessory.model.meta import MetaModel
from accessory.data.conversation import default_conversation

model = MetaModel.from_pretrained("/path/to/pretrained", max_seq_len=2048)

conv = default_conversation()

# for multi-turn-finetuned model
q1 = "What's the best programming language in the world?"
a1 = "The best programming language in the world in PHP."
q2 = "Are you sure? Why not Python?"
qas = [[q1, a1], [q2, None]]  # leave the last answer, namely the one to generate, to None
conv.load_qas(qas)
prompt = conv.get_prompt()
# prompt is equal to:
#   "A chat between a curious human and an artificial intelligence assistant. "
#   "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
#   "### Human: What's the best programming language in the world?\n"
#   "### Assistant: The best programming language in the world in PHP.\n"
#   "### Human: Are you sure? Why not Python?\n"
#   "### Assistant:"

# conv_sep is the symbol marking the end of one response, equal to "###" in this example
conv_sep = conv.response_end_signal

# ------EITHER-------
response = None
for response_in_progress in model.stream_generate(
        prompt, image=None, max_gen_len=512, additional_stop_symbols=[conv_sep]
):
    response = response_in_progress['text']
print(response)
# --------OR---------
response = None
for response_in_progress in model.stream_generate(prompt, image=None, max_gen_len=512):
    sep_pos = response_in_progress["text"].find(conv_sep)
    if sep_pos != -1:
        response = response_in_progress["text"][:sep_pos]
        break
    else:
        response = response_in_progress["text"]
print(response)
```
:::{important}
For pretrained and single-turn models, the end of the response is controlled by the generation of the `<EOS>` token.
In contrast, for multi-turn models, the end of the response is determined by template-specific seperator, *e.g.* `###`
in the example above. Since `MetaModel.generate` (only when batch size == 1) and `MetaModel.stream_generate` 
automatically halt when `<EOS>` is generated, for pretrained and single-turn models, nothing special about generation 
halting need to be taken care of. However, for the multi-turn case, the halting symbol need to be explicitly specified,
as we show in the example.
:::

#### Multi-modal Models
To inference with multi-modal models, you simply need to instantiate the `MetaModel` with `with_visual=True`, and
pass the image(s) to the generation function:

```python
from accessory.model.meta import MetaModel
from accessory.data.transform import get_transform
from PIL import Image

model = MetaModel.from_pretrained("/path/to/pretrained", with_visual=True, max_seq_len=2048)

image = Image.open("/path/to/image").convert("RGB")
transform_type = "padded_resize"  # or "resized_center_crop". Make it consistent across training & inference
transform = get_transform(transform_type, getattr(model.llma, 'image_size', 224))
image = transform(image).unsqueeze(0).cuda().bfloat16()

# ---------single turn---------
from accessory.data.system_prompt import format_prompt

prompt = format_prompt({"instruction": "What's in the image?"}, sys_name="alpaca")
response = None
for response_in_progress in model.stream_generate(prompt, image=image, max_gen_len=512):
    response = response_in_progress['text']
print(response)


# ---------multi turn---------
from accessory.data.conversation import default_conversation

qas = [["What's in the image?", None]]
conv = default_conversation()
conv.load_qas(qas)
prompt = conv.get_prompt()
conv_sep = conv.response_end_signal

response = None
for response_in_progress in model.stream_generate(
        prompt, image=image, max_gen_len=512, additional_stop_symbols=[conv_sep]
):
    response = response_in_progress['text']
print(response)
```

### Multi-GPU Inference with Model Parallelism
```python
from accessory.model.meta import MetaModel
from accessory.data.system_prompt import format_prompt

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
    model = MetaModel.from_pretrained("/path/to/pretrained", max_seq_len=2048,
                                      mp_group=dist.new_group(ranks=list(range(dist.get_world_size()))))

    instruction = "What's the best programming language in the world?"
    prompt = format_prompt({"instruction": instruction}, sys_name="alpaca")

    response = None
    for response_in_progress in model.stream_generate(prompt, image=None, max_gen_len=512):
        response = response_in_progress['text']
        print(response)


if __name__ == "__main__":
    N_GPU = 2
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

## Host Local Demos
We provide a series of scripts to host local gradio demos for easier interaction with trained LLaMA2-Accessory models.

:::{important}
As we have mentioned in [Prepare Checkpoints](#prepare-checkpoints), 
the `--llama_type``, --llama_config`, and `--tokenizer_path` arguments in the launching commands listed below can be
omitted as long as files recording the corresponding information exist under the path `--pretrained_path` point to.
:::

### Single-turn Single-modal Dialogue
Use the {link2repo}`[single_turn.py](accessory/demos/single_turn.py)` script for single-turn dialogues:

```bash
torchrun --nproc-per-node=$NPROC --master-port=$PORT demos/single_turn.py \
--pretrained_path $PRETRAINED --llama_type $LLAMA_TYPE --llama_config $LLAMA_CONFIG --tokenizer_path $TOKENIZER

# (Optional) Quantization-assistant Inference. To run on GPUs with limited VRAM, add the "--quant" flag.
# For example, less than 7GB of VRAM is required for the 7B model.
torchrun --nproc-per-node=$NPROC --master-port=$PORT demos/single_turn.py \
<--some_flags> --quant
```

### Single-turn Multi-modal Dialogue
Use the {link2repo}`[single_turn_mm.py](accessory/demos/single_turn_mm.py)` script for single-turn multi-modal dialogues:

```bash
torchrun --nproc-per-node=$NPROC --master-port=$PORT demos/single_turn_mm.py \
--pretrained_path $PRETRAINED --llama_type $LLAMA_TYPE --llama_config $LLAMA_CONFIG --tokenizer_path $TOKENIZER

# (Optional) Quantization-assistant Inference. To run on GPUs with limited VRAM, add the "--quant" flag.
# For example, less than 7GB of VRAM is required for the 7B model.
torchrun --nproc-per-node=$NPROC --master-port=$PORT demos/single_turn.py \
<--some_flags> --quant
```

### Multi-turn Single-modal Dialogue

For multi-turn single-modal dialogues, use the {link2repo}`[multi_turn.py](accessory/demos/multi_turn.py)` script:

```bash
python demos/multi_turn.py --n_gpus $NPROC \
--pretrained_path $PRETRAINED --llama_type $LLAMA_TYPE --llama_config $LLAMA_CONFIG --tokenizer_path $TOKENIZER

# (Optional) Quantization-assistant Inference. To run on GPUs with limited VRAM, add the "--quant" flag.
# For example, less than 7GB of VRAM is required for the 7B model.
python demos/multi_turn.py <--some_flags> --quant
```

### Multi-turn Multi-modal Dialogue

For multi-turn multi-modal dialogues, use the {link2repo}`[multi_turn_mm.py](accessory/demos/multi_turn_mm.py)` script:

```bash
python demos/multi_turn_mm.py --n_gpus $NPROC \
--pretrained_path $PRETRAINED --llama_type $LLAMA_TYPE --llama_config $LLAMA_CONFIG --tokenizer_path $TOKENIZER

# (Optional) Quantization-assistant Inference. To run on GPUs with limited VRAM, add the "--quant" flag.
# For example, less than 7GB of VRAM is required for the 7B model.
python demos/multi_turn_mm.py <--some_flags> --quant
```


## Model Zoo
```
├── convert
│   └── sg
│       ├── mixtral-8x7b-32kseqlen
│       ├── Falcon
│       ├── Falcon_180b
│       └── InternLM
└── finetune
    ├── mm
    │   ├── alpacaLlava_llamaQformerv2
    │   ├── alpacaLlava_llamaQformerv2_13b
    │   ├── alpacaLlava_llamaQformerv2Peft_13b
    │   ├── caption_llamaQformerv2
    │   ├── caption_llamaQformerv2_13b
    │   └── SPHINX
    │       ├── SPHINX
    │       ├── SPHINX-1k
    │       └── SPHINX-v2-1k
    └── sg
        ├── alpaca
        ├── alpaca_internLM_en
        ├── alpaca_internLM_zh
        ├── alpaca_llamaPeft_normBias
        ├── dialog_flan
        ├── dialog_lima
        ├── dialog_moss 
        ├── dialog_platypus
        ├── dialog_sharegpt
        ├── dialog_sharegpt_70b
        ├── dialog_ultra
        ├── dialog_wizardcode
        ├── dialog_wizardcode_codellama
        ├── dialog_wizardcode_loadcode220k
        ├── dialog_wizardLM
        └── gorilla
```

### How to Apply Delta Weights (Outdated)

:::{warning}

This section may be outdated as we have now released the full-version (i.e. merged) pretrained weights directly. Applying delta is no longer needed.

:::

We release checkpoints as delta weights to comply with the LLaMA2 model license. To use our provided weights for inference or further tuning, please first add our delta to the original LLaMA2 weights to obtain the full weights:

Instructions:

1. After agreeing to the License, Acceptable Use Policy, and Meta's privacy policy, proceed to download the LLaMA2 weights from [here](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).
2. Utilize the following scripts to obtain finetuned weights by applying our delta. Make sure to download the delta weights from the [model release page](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory).

For those who wish to download smaller models like peft, we have retained the delta weights. Simply add the `--down_diff` argument during download to facilitate the process.
   ```bash
   # For Download
   python tools/download.py  --model_name check/in/release/page --input_type sg/or/mm --output_path path/to/save --model_size 7B/13B/70B --down_config --down_diff
   # For Merging
   python tools/weight_operate.py  --pretrained_path /path/to/llama2/ --delta_path /path/to/delta --output_path /path/to/finetuned
   # For Separation
   python tools/weight_operate.py  --pretrained_path /path/to/llama2/ --delta_path /path/to/finetuned --output_path /path/to/delta --operate_type extract
   ```



