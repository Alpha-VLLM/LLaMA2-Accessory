# Installation Guide
**1. Basic setup**

```bash
# Create a new conda environment named 'accessory' with Python 3.10
conda create -n accessory python=3.10 -y
# Activate the 'accessory' environment
conda activate accessory
# Install required packages from 'requirements.txt'
pip install -r requirements.txt
```
**2. Optional: Install Flash-Attention**

LLaMA2-Accessory is powered by [flash-attention](https://github.com/Dao-AILab/flash-attention) for efficient attention computation. 

```bash
pip install flash-attn --no-build-isolation
```

*We highly recommend installing this package for efficiency*. However, if you have difficulty installing this package, LLaMA2-Accessory should still work smoothly without it.

**3. Optional: Install Apex**

LLaMA2-Accessory also utilizes [apex](https://github.com/NVIDIA/apex), which needs to be compiled from source. Please follow the [official instructions](https://github.com/NVIDIA/apex#from-source) for installation.

**We strongly advise installing this package for optimal performance*. However, if you have difficulty installing this package, LLaMA2-Accessory can operate smoothly without it.