# Frequently Asked Questions (FAQ)

## 1. What Is Flash Attention?

Flash Attention is a module designed to enhance the efficiency of attention computation. You may install it using the command below:

```bash
pip install flash-attn --no-build-isolation
```

Please note that the `flash_attn` module is *not* supported on all types of GPUs. Should it not be applicable to your machine, kindly set `USE_FLASH_ATTENTION` to `False` in [accessory/configs/global_configs.py](../accessory/configs/global_configs.py). The vanilla attention computation will then be utilized.

## 2. Requirements for Apex and CUDA Version

This project relies on [Apex](https://github.com/NVIDIA/apex), which necessitates compilation from source. We are planning to make it an optional choice in the future, but currently, you are advised to follow the [official instructions](https://github.com/NVIDIA/apex#from-source). Please ensure your CUDA version (**11.7**) aligns with the requirements for Apex.

## 3. How to Apply Delta Weights?

**(Please note that the following content may be outdated as we have now fully open-sourced our pre-trained weights)**

We release our checkpoints as delta weights to comply with the LLaMA2 model license. To utilize our provided weights for inference or further tuning, kindly adhere to the following instructions to merge our delta into the original LLaMA2 weights to obtain the full weights:

1. Upon agreeing to the License, Acceptable Use Policy, and Meta's privacy policy, proceed to download the LLaMA2 weights from [here](link).
2. Utilize the following scripts to obtain finetuned weights by applying our delta. Make sure to download the delta weights from the model release page.
    ```bash
    # For Download
    python tools/download.py --model_name check/in/release/page --input_type sg/or/mm --output_path path/to/save --model_size 7B/13B/70B --down_config
    # For Merging
    python tools/weight_operate.py --pretrained_path /path/to/llama2/ --delta_path /path/to/delta --output_path /path/to/finetuned
    ```





***Should you have any further queries, please don't hesitate to post in the issue section. We will endeavor to respond promptly to your questions. Thank you for engaging with our project.***
