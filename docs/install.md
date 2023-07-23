# Environment Setup
1. Setup up a new conda env and install required packages
    ```bash
    # create conda env
    conda create -n accessory python=3.10 -y
    conda activate accessory
    # install packages
    pip install -r requirements.txt
    ```
2. This project relies on [apex](https://github.com/NVIDIA/apex), which needs to be compiled from source. Please follow the [official instructions](https://github.com/NVIDIA/apex#from-source).
3. LLaMA2-Accessory is powered by [flash-attention](https://github.com/Dao-AILab/flash-attention) for efficient attention computation:
    ```bash
    pip install flash-attn --no-build-isolation
    ```
   Note that the `flash_attn` module is *not* supported on all types of GPUs. If it is not applicable on your machine, please set `USE_FLASH_ATTENTION` in [accessory/configs/global_configs.py](../accessory/configs/global_configs.py) to `False`. Vanilla attention computation will then be used.