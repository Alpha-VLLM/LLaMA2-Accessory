# Evaluation

## Evaluation with OpenCompass
[OpenCompass](https://github.com/open-compass/opencompass) is a one-stop platform for large model evaluation. It 
supports a wide range of evaluation datasets and is well maintained. OpenCompass has inherently implemented an 
[interface](https://github.com/open-compass/opencompass/blob/main/opencompass/models/accessory.py) for LLaMA2-Accessory
models, so that all the models developed using LLaMA2-Accessory can be seamlessly evaluated on all the datasets 
supported by OpenCompass.

### Background
Before evaluation, please briefly read the [OpenCompass document](https://opencompass.readthedocs.io/) to understand
its basic workings. Generally speaking, for each time of evaluation, opencompass expects a pair of data and model 
configurations. The data configuration specifies the datasets used for the evaluation, and the model configuration
determines which model will be evaluated. In most cases, you only need to work on these two configuration files to
evaluate your own model.

### Configuration
There is no difference in the data configuration when testing llama2-accessory models compared to testing other models,
and you may refer to the OpenCompass document to learn how to customize it. For model configuration, some
off-the-shelf examples for typical LLaMA2-Accessory models are provided 
[here](https://github.com/open-compass/opencompass/tree/main/configs/models/accessory). Users are encouraged to read
and modify these example configurations to meet their own requirements.

### Running Evaluation
The evaluation launching command for LLaMA2-Accessory models is also totally the same as that for other models, so you
can use what you learned from the OpenCompass document without special care. For convenience, we provide the command
we use below:

::::{tab-set}

:::{tab-item} Local
```bash
#!/bin/bash

model="accessory_mixtral_8x7b"  # will use configs/models/accessory/accessory_mixtral_8x7b.py
dataset="base_small"  # will use configs/datasets/collections/base_small.py

python run.py --models ${model} --datasets ${dataset} \
--work-dir ./outputs/"${exp_name}"  --config-dir ./configs --reuse
```
:::

:::{tab-item} Slurm
```bash
#!/bin/bash

model="accessory_mixtral_8x7b"  # will use configs/models/accessory/accessory_mixtral_8x7b.py
dataset="base_small"  # will use configs/datasets/collections/base_small.py
exp_name="some_name_you_like"
partition="your_slurm_partition" 

python run.py --slurm -p ${partition} -q auto --models ${model} --datasets ${dataset} \
--work-dir ./outputs/"${exp_name}"  --config-dir ./configs --reuse
```
:::

::::


:::{note}
The `base_small.py` data configuration consists a lot of datasets, and you may not want to test them all. 
Please read this file, comment the unneeded datasets, and possibly add some other datasets you want before evaluation.
:::