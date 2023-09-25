# Inference
## Model Zoo
```
├── convert
│   └── sg
│       ├── Falcon
│       ├── Falcon_180b
│       └── InternLM
└── finetune
    ├── mm
    │   ├── alpacaLlava_llamaQformerv2
    │   ├── alpacaLlava_llamaQformerv2_13b
    │   ├── alpacaLlava_llamaQformerv2Peft_13b
    │   ├── caption_llamaQformerv2
    │   └── caption_llamaQformerv2_13b
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
The models are availabel at [🤗Hugging Face](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory).

Get started with our project effortlessly! Simply run the provided startup script from the `accessory` of the project. This will guide you through an interactive setup, making it a breeze to configure and run the model.

- **For Linux users:**

  ```bash
  bash demos/start.sh
  ```

- **For Windows users:**
(not recommended)
  ```powershell
  .\demos\start.bat
  ```

Enjoy your journey with our project!


## Prerequisites

Before running the inference code, users must ensure that they have correctly installed and configured all necessary environments according to the instructions in the [Installation Document](./install.md).

### How to Download Pre-train Weights

**We are pleased to announce that we have now released the full-version (i.e. merged) pre-trained weights**. You can directly download and utilize them without the need to merge original and delta weights. This simplifies the downloading process and provides an immediate user experience. 

:::{important}

The usage of our released checkpoints should comply with the base LLM's model license: [LLaMA](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md).

:::

Example commands for download are as follows:

```bash
# Now support interactive input for all parameters
python tools/download.py
# It can also be directly specified in the command line.
python tools/download.py --model_name check/in/model/zoo --input_type sg/or/mm --output_path path/to/save --model_size 7B/13B/34B/70B [--down_config] [--down_internLM] [--down_code]
```

Please continue to stay updated with our latest releases and feel free to share your needs and feedback with us.

### How to Apply Delta Weights (Outdated)

:::{warning}

This section may be outdated as we have now released the full-version (i.e. merged) pre-trained weights directly. Applying delta is no longer needed.

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



## Inference Scenarios

Below are examples of how to run inference in three different scenarios: single-turn dialogue, multi-turn dialogue, and multi-modal dialogue.

### Single-turn Dialogue

Use the `single_turn.py` script for single-turn dialogues:

```bash
python demos/single_turn.py \
--llama_config /path/to/params.json --tokenizer_path /path/to/tokenizer.model \
--pretrained_path /path/to/finetuned

# (Optional) Quantization-assistant Inference. To run on GPUs with limited VRAM, add the "--quant" flag.
# For example, less than 7GB of VRAM is required for the 7B model.
python demos/single_turn.py <--some_flags> --quant
```

### Multi-turn Dialogue

For multi-turn dialogues, use the `multi_turn.py` script:

```bash
python demos/multi_turn.py \
--llama_config /path/to/params.json --tokenizer_path /path/to/tokenizer.model \
--pretrained_path /path/to/finetuned

# (Optional) Quantization-assistant Inference. To run on GPUs with limited VRAM, add the "--quant" flag.
# For example, less than 7GB of VRAM is required for the 7B model.
python demos/multi_turn.py <--some_flags> --quant
```

### Multi-modal Dialogue

And for multi-modal dialogues, use the `single_turn_mm.py` script:

```bash
torchrun --nproc-per-node=2  demos/single_turn_mm.py \
--llama_config /path/to/params.json --tokenizer_path /path/to/tokenizer.model \
--pretrained_path /path/to/multimodel_llama

# (Optional) Quantization-assistant Inference. To run on GPUs with limited VRAM, add the "--quant" flag.
# For example, less than 7GB of VRAM is required for the 7B model.
torchrun --nproc-per-node=1  demos/single_turn_mm.py <--some_flags> --quant
```

Please replace `/path/to/params.json`, `/path/to/tokenizer.model` and `/path/to/finetuned` or `/path/to/multimodel_llama` with your actual file paths.
