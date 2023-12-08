# Language-only Parameter-Efficient Finetuning

## Bias&Norm Tuning of LLaMA2-7B on Alpaca 

**Script:**

+  {link2repo}`[exps/finetune/sg/alpaca_llamaPeft_normBias.sh](accessory/exps/finetune/sg/alpaca_llamaPeft_normBias.sh)`

**Data:**

+ [💾alpaca_gpt4_data.json](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json)

**Model Release:**

+ [🤗checkpoint](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/blob/main/finetune/sg/alpaca_llamaPeft_normBias/consolidated.00-of-01.model.pth)

**Host Local Demo:**

```bash
torchrun --nproc-per-node=1  demos/single_turn.py \
--llama_type llama_peft \
--llama_config /path/to/params.json --tokenizer_path /path/to/tokenizer.model \
--pretrained_path /path/to/alpaca_finetuned
```

## Bias&Norm&LoRA Tuning of LLaMA2-7B on Alpaca

**Script:**

+ {link2repo}`[exps/finetune/sg/alpaca_llamaPeft_normBiasLora.sh](accessory/exps/finetune/sg/alpaca_llamaPeft_normBiasLora.sh)`

*Explanation*: This experiment assigns two filenames to `llama_config` simultaneously. The first filename, like most other experiments, points to the `params.json` file released by META that distinguishes model sizes. {link2repo}`[The second filename](accessory/configs/model/finetune/sg/llamaPeft_normBiasLora.json)`, on the other hand, defines the inner dimension of LoRA. Through this separated design, one may simply change the first filename to switch to other model sizes, without the need to create new model configuration files.

**Data:**

+ [💾alpaca_gpt4_data.json](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json)

**Host Local Demo:**

```bash
torchrun --nproc-per-node=1  demos/single_turn.py \
--llama_type llama_peft \
--llama_config /path/to/params.json configs/model/finetune/sg/llamaPeft_normBiasLora.json \
--tokenizer_path /path/to/tokenizer.model \
--pretrained_path /path/to/alpaca_finetuned
```

Note that `--llama_config` should be consistent with training, i.e. include both configuration files.

## LLaMA-Adapter of LLaMA2-7B on Alpaca 

**Script:**

+ {link2repo}`[exps/finetune/sg/alpaca_llamaAdapter.sh](accessory/exps/finetune/sg/alpaca_llamaAdapter.sh)`

**Data:**

+ [💾alpaca_gpt4_data.json](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json)

**Host Local Demo:**

```bash
torchrun --nproc-per-node=1  demos/single_turn.py \
--llama_type llama_adapter \
--llama_config /path/to/params.json --tokenizer_path /path/to/tokenizer.model \
--pretrained_path /path/to/alpaca_finetuned
```

## Bias&Norm&LoRA Tuning of LLaMA2-7B on Multi-turn ShareGPT

**Script:**

+ {link2repo}`[exps/finetune/sg/dialog_sharegpt_llamaPeft_normBiasLora.sh](accessory/exps/finetune/sg/dialog_sharegpt_llamaPeft_normBiasLora.sh)`

**Data:**

+ Please collect and process the data on your own. {link2repo}`[Here](data_example/ShareGPT.json)` is a toy example showing the proper format of the data file.

**Host Local Demo:**

```bash
python demos/multi_turn.py \
--llama_type llama_peft \
--llama_config /path/to/params.json configs/model/finetune/sg/llamaPeft_normBiasLora.json \
--tokenizer_path /path/to/tokenizer.model \
--pretrained_path /path/to/sharegpt_finetuned
```

---

*More use cases coming soon...*