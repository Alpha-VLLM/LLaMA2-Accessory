# Pretrain
LLaMA2-Accessory currently supports two kinds of pretraining datasets: the *vanilla* dataset and the *packed* dataset. Which one is used for training is controlled by the `--packed_data` argument in `main_pretrain.py`.

## Vanilla Dataset

The vanilla dataset is supported in {link2repo}`[data/falcon.py](accessory/data/falcon.py)`. It loads data directly from `.parquet` data files (as an example, see [*Falcon Refined-web*](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)). With the vanilla dataset, every piece of data will be converted into tokens of  fixed length. Specifically, it will be truncated if it is longer than the target length, and padded if shorter.

An example for pretraining with the vanilla dataset is provided in {link2repo}`[exps/pretrain/vanilla.sh](accessory/exps/pretrain/vanilla.sh)`. Here are some notes about the script:

1\. To run the script on your own machine, point the `llama_config` variable to the `*.json` files defining model configuration, and the `tokenizer_path` variable to the spm (*i.e.* tokenizer.model) or huggingface tokenizer.
:::{tip}

See FAQ to know more about [llama_config](./faq.md#how-to-set-llama_config) and [tokenizer_path](./faq.md#how-to-set-tokenizer_path).

:::

2\. A meta file specifying the list of `.parquet` files to use should be created and pointed to by the `data_meta_path` variable. We provide an example meta file for the *Falcon Refined-web* dataset {link2repo}`[here](data_example/PretrainMeta.json)`.
  + The elements in the meta file should be either absolute paths, or paths relative to `data_root`.

## Packed Dataset

For more efficient token utilization, the packed dataset is supported in {link2repo}`[data/falcon_packed.py](accessory/data/falcon_packed.py)`. The packed dataset concatenates contents from different data pieces into a whole and then splits it into equal-length segments. To train with packed dataset, you need to first pre-process your data, namely to tokenize, concatenate, split, and save them. A script for doing this is provided in {link2repo}`[tools/generate_packed_data.py](accessory/tools/generate_packed_data.py)`.

```bash
python -u tools/generate_packed_data.py
```

An example for pretraining with the packed dataset is provided in {link2repo}`[exps/pretrain/13B_packed.sh](accessory/exps/pretrain/13B_packed.sh)`. Similar to the case of the vanilla dataset, you also need to create a meta file and point `data_meta_path` to it. If you use our `generate_packed_dataset.py` to preprocess data, elements in the meta file should end with `.pkl` (See {link2repo}`[here](data_example/PretrainMetaPacked.json)` for example). 



