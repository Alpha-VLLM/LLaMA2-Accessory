# Prerequisites

To run our provided experiment scripts on you own machine, please first adjust the following configurations:

+ Modify the value of the `pretrained_path` variable in the `.sh` file. This variable should point to the directory containing checkpoints to finetune from.

 + If you finetune from the officianl LLaMA / LLaMA2 checkpoints released by META, the directory should be like:
   ```
   pretrained_path
   ├── consolidated.00.pth
   ├── consolidated.01.pth
   └── ...
   ```

   and your should set `pretrained_type=meta_ori` in the `.sh` file. 
   
   Alternatively, you may also finetune from checkpoints saved by LLaMA2-Accessory. In such cases, the directory should be like:
   
   ```
   pretrained_path
   ├── consolidated.00-of-**.model.pth
   ├── consolidated.01-of-**.model.pth
   └── ...
   ```

   and your should set `pretrained_type=consolidated` in the `.sh` file

+ Point `llama_config` in `.sh` scripts to the model configuration files (`*.json`) that specify model size 
  (7B, 13B, ...) and other settings (if any). See [here](../faq.md#how-to-set-llama_config) to know more.
+ Point `tokenizer_path` in `.sh` to the tokenizer, See more [here](../faq.md#how-to-set-tokenizer_path).
+ Point the `data_config` argument in `.sh` to a `.yaml` file defining the collection of finetuning datasets, 
  each of which is identified by a `.json` meta file. 
+ Modify *model parallel size* properly. 'model parallel size' specifies how the parameters of each complete model 
  are split and distributed across multiple GPUs. The Meta official has provided a set of corresponding relationships,
  for example, 7B corresponds to a model parallel size of 1, 13B corresponds to 2, and 70B corresponds to 8. 
  The effect of this is to keep the load on each GPU relatively constant as the total number of model parameters
  increases. Overall, following this guideline is generally a good choice in most situations; however, if you are 
  very familiar with the subject, you can also try to break this binding.

:::{important}

LLaMA2-Accessory itself supports model parallelism (which, within the current scope of LLaMA2-Accessory, is equivalent 
to tensor parallelism) and Fully Sharded Data Parallel (FSDP). Both of these involve the partitioning of the model, 
but it is important to note that they are **very different and orthogonal** (i.e., they can be used simultaneously) 
technologies. A brief understanding of these two technologies is very helpful for better utilizing LLaMA2-Accessory. 
[This blog from Microsoft](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)
is an excellent learning resource.

:::
