# Installation Guide

## 1. Basic Setup

```bash
# Create a new conda environment named 'accessory' with Python 3.10
conda create -n accessory python=3.10 -y
# Activate the 'accessory' environment
conda activate accessory
# Install required packages from 'requirements.txt'
pip install -r requirements.txt
```
## 2. Optional: Install Flash-Attention

LLaMA2-Accessory is powered by [flash-attention](https://github.com/Dao-AILab/flash-attention) for efficient attention computation. 

```bash
pip install flash-attn --no-build-isolation
```

*We highly recommend installing this package for efficiency*. However, if you have difficulty installing this package, LLaMA2-Accessory should still work smoothly without it.

## 3. Optional: Install Apex
:::{danger}

While Apex can bring some efficiency improvement, it is not necessary to make LLaMA2-Accessory work. 

Note that LLaMA2-Accessory works smoothly with either:
1. Apex not installed at all; OR
2. Apex successfully installed with CUDA and C++ extensions.

However, it will fail when:
1. A Python-only build of Apex is installed.

Therefore, if errors like `No module named 'fused_layer_norm_cuda'` are reported, it generally means that you are 
using a Python-only Apex build. Please run `pip uninstall apex` to remove the build, after which LLaMA2-Accessory
should be able to work.
:::

LLaMA2-Accessory also utilizes [apex](https://github.com/NVIDIA/apex), which needs to be compiled from source. Please follow the [official instructions](https://github.com/NVIDIA/apex#from-source) for installation. 
Here are some tips based on our experiences:

**Step1**: Check the version of CUDA with which your torch is built:
 ```python
# python
import torch
print(torch.version.cuda)
```
If you have followed the instructions in [Basic Setup](#1-basic-setup) to build your environment from `requirements.txt`, you should see cuda version `11.7`. 

**Step2**: Check the CUDA toolkit version on your system:
```bash
# bash
nvcc -V
```
**Step3**: If the two aforementioned versions mismatch, or if you do not have CUDA toolkit installed on your system,
please download and install CUDA toolkit from [here](https://developer.nvidia.com/cuda-toolkit-archive) with version matching the torch CUDA version.   

:::{tip}

Note that multiple versions of CUDA toolkit can co-exist on the same machine, and the version can be easily switched by changing the `$PATH` and `$LD_LIBRARY_PATH` environment variables. 
So there is no need to worry about your machine's environment getting messed up.

:::

**Step4**: You can now start installing apex:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```


## 4. Install LLaMA2-Accessory as Python Packege
After going through the previous steps, you can now use most of the functionalities provided by LLaMA2-Accessory, including pretraining, finetuning, etc. 
However, the usage is restricted by working directory. For example, it would be inconvenient to instantiate LLaMA2-Accessory models in other projects.
To solve this problem, you can install LLaMA2-Accessory into your python enviroment as a package:
```bash 
# bash
# go to the root path of the project
cd LLaMA2-Accessory
# install as package
pip install -e .
```
After this, you can invoke `import accessory` and `import SPHINX` anywhere in you machine, without the restriction of working directory.



