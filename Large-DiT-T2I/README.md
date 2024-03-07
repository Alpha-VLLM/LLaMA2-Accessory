# Text-to-Image Large Diffusion Transformer 🎆

We release the ***Text-to-Image Large Diffusion Transformer*** (**L-DiT-T2I-3B**), inspired by the methodologies of [LLaMA](https://github.com/facebookresearch/llama) and [DiT](https://github.com/facebookresearch/DiT). 

![image-20240307160444196](assets/sample.png)

## Introduction

**TL; DR:** Large-DiT-T2I is a text-to-image diffusion model building upon [Large-DiT](https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/main/Large-DiT-ImageNet) that can generate high-quality artistic images.  Large-DiT-T2I is equipped with zero-initialization attention for arbitrary text-conditioning and unique “newline” and “pad” tokens to generate images with arbitrary resolution. 

## Model Zoo

The checkpoints of our model will be released soon~

| Resolution | Download URL   |
| ---------- | -------------- |
| 256        | To be released |
| 512        | To be released |
| 1024       | To be released |

## Installation

For installation instructions, please refer to the [Large-DiT installation guide](https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/main/Large-DiT-ImageNet).

## Sampling

Once the necessary checkpoints are prepared, you can initiate a demo on your local machine using Gradio. Run the following command to get started:

```bash
# You can change the --ckpt to different resolution models
python -u demo.py --ckpt /path/to/trained/ckpt
```

## Training

1. Prepare the dataset. The training datasets, including the paths to image samples and the corresponding text prompts, are expected to be collected in `*.json` files:

   ```
   # example_data.json
   [
     {
       "path": "/path/to/image1",
       "prompt": "image1 prompt"
     },
       {
       "path": "/path/to/image2",
       "prompt": "image2 prompt"
     },
     ...
   ]
   ```

   Suppose the above file is named `example_data.json`, it should then be recorded in a `*.yaml` file, which includes all the `*.json` datasets that you expect to use:

   ```yaml
   META:
     - path: '/path/to/example_data.json'
   #  if the concatenation of multiple datasets is needed
   #  - path: 'path/to/example_data2.json'
   #  - path: 'path/to/example_data3.json'
   ```

   In our experiments, we use the [JourneyDB](https://journeydb.github.io/) dataset. You may refer to [configs/data/JourneyDB.yaml](./configs/data/JourneyDB.yaml) as an example.

1. Example training scripts are provided in the [exps](./exps) folder. Configure the `*.yaml` data collection file correctly before running. The scripts are for 8xA100-80G machines, and you may need to reduce `--micro_batch_size` or model size in case of OOM.

## Acknowledgements

The codebase is extended from [DiT](https://github.com/facebookresearch/DiT) and [LLaMA](https://github.com/facebookresearch/llama). This project is also highly motivated by [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha) and [JourneyDB](https://journeydb.github.io/). Thanks for their awesome work! 

## Hiring Announcement
🔥 We are hiring interns, postdocs, and full-time researchers at the General Vision Group, Shanghai AI Lab, with a focus on large vision-language model and large diffusion model. If you are interested, please contact gaopengcuhk@gmail.com.