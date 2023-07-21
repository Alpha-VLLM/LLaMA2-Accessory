# Fine-tuning
This document demonstrates the fine-tuning use cases supported by LLaMA2-Accessory

> ## Prerequisites
> To run our provided experiment scripts on you own machine, please first adjust the following configurations:
> + Modify the value of the `pretrained_path` variable in the `.sh` file. This variable should point to the directory containing checkpoints to fine-tune from.
>   + If you fine-tune from the officianl LLaMA / LLaMA2 checkpoints released by META, the directory should be like:
>       ```
>       pretrained_path
>       ├── consolidated.00.pth
>       ├── consolidated.01.pth
>       └── ...
>       ```
>     and your should set `pretrained_type=meta_ori` in the `.sh` file.
>   + Alternatively, you may also fine-tune from checkpoints saved by LLaMA2-Accessory. In such cases, the directory should be like:
>       ```
>       pretrained_path
>       ├── consolidated.00-of-**.model.pth
>       ├── consolidated.01-of-**.model.pth
>       └── ...
>       ```
>    and your should set `pretrained_type=consolidated` in the `.sh` file.
> + The `data_config` argument in the `.sh` file points to a `.yaml` file defining the fine-tuning datasets. You need to first download the data and modify the paths in `.yaml` to the correct location.


## Full-Parameter Fine-tuning

---

### Single-turn instruction-tuning of LLaMA2-7B on alpaca
Script: 
+ [exps/finetune/sg/alpaca.sh](../accessory/exps/finetune/sg/alpaca.sh)

Data:
+ https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json

Model Release:
+ [checkpoint]()

---

### Multi-turn-dialog fine-tuning of LLaMA2-7B on ShareGPT
Script:
+ [exps/finetune/sg/dialog_sharegpt.sh](../accessory/exps/finetune/sg/dialog_sharegpt.sh)

Data:
+ Please collect and process the data on your own. [Here](../data_example/ShareGPT.json) is a toy example showing the proper format of the data file.

Model Release:
+ [checkpoint]()

---

*More use cases coming soon...*