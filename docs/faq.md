# Frequently Asked Questions (FAQ)

## Install & Preperation
### Why are my LLaMA checkpoints `*.bin` or `*.safetensors`?
There are two common formats for LLaMA checkpoints: the original format provided by Meta, and the 
Huggingface format. Checkpoints in the former format are stored in `consolidated.*.pth` files, whereas
checkpoints in the latter format are stored in `*.bin` or `*.safetensors` files. **LLaMA2-Accessory works 
with the former format (`consolidated.*.pth` files).**


## Model
### How to set `llama_config`?
In LLaMA2-Accessory, each model class has a corresponding `ModelArgs` class, which is defined in the same file as the model class. 
The `ModelArgs` class contains arguments for configuring of the model. An example of `ModelArgs` can be found in 
{link2repo}`[this file](accessory/model/LLM/llama.py)`. Arguments in `ModelArgs` can be given default values at their definition. 
On the other hand, the overriding of the arguments can be achieved by filling the `llama_config` argument when 
creating `MetaModel`.

`llama_config` is expected to be a list of strings, specifying the paths to the `*.json` configuration files. 
The most commonly used configuration files are those defining model sizes (7B, 13B, 65B, *etc.*), **which are officially 
provided by Meta and named `params.json`**. For example, the configuration file for 13B llama is 
provided at `https://huggingface.co/meta-llama/Llama-2-13b/blob/main/params.json`. So generally when you want to change 
the model from `7B` to `13B` while leaving other things consistent, you can simply change `llama_config` from 
`['/path/to/7B/params.json']` to `['/path/to/13B/params.json']`

Except model size, there are still other things to configure and can be different from model to model.
For example, the PEFT model {link2repo}`[llama_peft](accessory/model/LLM/llama_peft.py)` allows users to 
configure the detailed PEFT settings, including the rank of lora and whether to tune bias.
{link2repo}`[llamaPeft_normBiasLora.json](accessory/configs/model/finetune/sg/llamaPeft_normBiasLora.json)` 
contains the configuration that we usually use:
```json
{"lora_rank": 16, "bias_tuning": true}
```
Based on this, when instantiating a `llama_peft` model, we can set `llama_type=llama_peft`, and 
`llama_config = ['/path/to/7B/params.json', '/path/to/llamaPeft_normBiasLora.json']` for 7B model, and 
`llama_config = ['/path/to/13B/params.json', '/path/to/llamaPeft_normBiasLora.json']` for 13B model. Of course, you can 
also merge the size and PEFT configs into a single file, and the effect is the same.

:::{note} 

When multiple `.json` config files are assigned to `llama_config`, The combined configuration from all these files
will be used, with keys from later files overwriting those from earlier ones. This is especially handy when you want
to make specific model configuration adjustments, like the LoRA dimension, which is consistent across various model
sizes, eliminating the need to produce individual files for each size.

:::

Note that the following arguments in `ModelArgs` are relatively special and their final values are not determined by 
the specification in `llama_config`:
+ `max_seq_len`: `MetaModel.__init__` receives an argument with the same name, which directly determines the value
+ `max_batch_size`: is currently hard-coded to be 32 in `MetaModel.__init__`
+ `vocab_size` is dynamically determined by the actual vocabulary size of the tokenizer



### How to set `tokenizer_path`?
LLaMA2-Accessory supports both spm tokenizers (provided by Meta, generally named `tokenizer.model`) and huggingface
tokenizers (composed of `tokenizer.json` and `tokenizer_config.json`). When using spm tokenizers, 
`tokenizer_path` should point to the `tokenizer.model` file; when using huggingface tokenizers, 
`tokenizer_path` should point to **the directory** containing `tokenizer.json` and `tokenizer_config.json`.

:::{tip}

For the LLaMA family, the tokenizer is the same across LLaMA and LLaMA2, and across different model sizes (in most 
cases the `tokenizer.model` file is downloaded together with LLaMA weights; you can also download it separately from
[here](https://huggingface.co/meta-llama/Llama-2-13b/blob/main/tokenizer.model)). In contrast, CodeLLaMA uses a
different [tokenizer](https://huggingface.co/codellama/CodeLlama-7b-hf/blob/main/tokenizer.model).

:::

---


***Should you have any further queries, please don't hesitate to post in the issue section. We will endeavor to respond promptly to your questions. Thank you for engaging with our project.***
