# Prerequisites

To run our provided experiment scripts on you own machine, please first adjust the following configurations:

+ Modify the value of the `pretrained_path` variable in the `.sh` file. This variable should point to the directory containing checkpoints to fine-tune from.

 + If you fine-tune from the officianl LLaMA / LLaMA2 checkpoints released by META, the directory should be like:
   ```
   pretrained_path
   ├── consolidated.00.pth
   ├── consolidated.01.pth
   └── ...
   ```

   and your should set `pretrained_type=meta_ori` in the `.sh` file. 
   
   Alternatively, you may also fine-tune from checkpoints saved by LLaMA2-Accessory. In such cases, the directory should be like:
   
   ```
   pretrained_path
   ├── consolidated.00-of-**.model.pth
   ├── consolidated.01-of-**.model.pth
   └── ...
   ```

   and your should set `pretrained_type=consolidated` in the `.sh` file

+ Point `llama_config` in `.sh` to the model parameter file (usually named as `params.json`) that differentiates model sizes (7B, 13B, ...)

  + You can also assign multiple `.json` config files to `llama_config` simultaneously.  The combined configuration from all these files will be used, with keys from later files overwriting those from earlier ones. This is especially handy when you want to make specific model configuration adjustments, like the LoRA dimension, which is consistent across various model sizes, eliminating the need to produce individual files for each size.

+ Point `tokenizer_path` in `.sh` to the tokenizer file (usually named as `tokenizer.model`)
+ The `data_config` argument in the `.sh` file points to a `.yaml` file defining the fine-tuning datasets. You need to first download the data and modify the paths in `.yaml` to the correct location. 
+ LLaMA models of different model sizes use different *model parallel sizes*. For example, LLaMA 7B's default model parallel size is 1, and LLaMA 13B's default model parallel size is 2. Thus, if you want to modify a script from training 7B models to training 13B models, the model_parallel parameter should be modified accordingly.

  + If you just train your model from scratch (i.e. do not load any pre-trained checkpoints), you may change the model parallel size arbitrarily based on your needs. However, if you do load some pre-trained checkpoints, it is important to guarantee the same model parallel size as pre-trained checkpoints. 

