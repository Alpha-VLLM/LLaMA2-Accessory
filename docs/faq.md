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

## 4. How to Utilize Trainable Parameters from `main_finetune.py`?

### **Merging with LLaMA2 Weights**

If you've only saved the trainable parameters from `main_finetune.py` and wish to combine them with the `llama2` weights:

- Use our recently introduced feature where demo Python files accept multiple directories for the `--pretrained_path` argument. The function `util.tensor_parallel.load_tensor_parallel_model_list` will autonomously discern checkpoint types (e.g., "meta_ori", "consolidated", or "diff") based on their names and load them sequentially. If a parameter appears in multiple checkpoints, later ones will overwrite earlier ones.
  
- If you prefer manual merging:
  ``` python
  for key, value in new_state_dict.items():
      ori_state_dict[key] = value
  ```
  Remember, compared to the original llama checkpoints, our parameter names have an added `"llma."` prefix.

### **Using the Parameters for Demos**

For those who saved both trainable and non-trainable parameters from `main_finetune.py`:

- They can be directly used in the `single_turn_mm.py` demo. Ensure `--only_save_trainable` is set to `False`. Here's how to use the saved weights:
```bash
python demos/single_turn_mm.py --pretrained_path <folder/to/base/model> <folder/to/trainable/parameters> <--other_flags>
```

---





***Should you have any further queries, please don't hesitate to post in the issue section. We will endeavor to respond promptly to your questions. Thank you for engaging with our project.***
