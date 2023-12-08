# Finetuning with Quantization

We support <u>Q</u>uantized <u>P</u>arameter-<u>E</u>fficient <u>F</u>ine-<u>T</u>uning (**QPEFT**) methods including **QNormBias** and **QNormBiasLoRA**, which significantly minimize the computing demands. In QPEFT, we quantize the base model while only retain carefully selected trainable parameters.

- **QNormBias**. Only the bias term and normalization weights are allowed for gradient updates. The pretrained LLaMA2 weights are quantized and frozen.

- **QNormBiasLoRA**. The bias term, LoRA weights, and  normalization weights are allowed for gradient updates. The pretrained LLaMA2 weights are quantized and frozen.

## Best Practice

```bash
# Enable quantization with flag "--quant" and "--only_save_trainable"
torchrun <--some_flags> main_finetune.py <--some_flags> \
--quant --only_save_trainable
```

For more details, please check the following scripts:

| Method        | Finetune <u>Language-only</u> LLaMA 2                                                                                                                         | Finetune <u>Multi-Modal</u> LLaMA 2                                                                                                                                     |
|:-------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| QNormBias     | [alpaca_llamaPeft_normBias_QF.sh](https://github.com/Alpha-VLLM/LLaMA2-Accessory/blob/main/accessory/exps/finetune/sg/alpaca_llamaPeft_normBias_QF.sh)         | -                                                                                                                                                                        |
| QNormBiasLoRA | [alpaca_llamaPeft_normBiasLora_QF.sh](https://github.com/Alpha-VLLM/LLaMA2-Accessory/blob/main/accessory/exps/finetune/sg/alpaca_llamaPeft_normBiasLora_QF.sh) | [alpacaLlava_llamaQformerv2Peft_QF_13B.sh](https://github.com/Alpha-VLLM/LLaMA2-Accessory/blob/main/accessory/exps/finetune/mm/alpacaLlava_llamaQformerv2Peft_QF_13B.sh) |

## Comparison

Models can be loaded in 4-bit NormalFloat (NF4) data format which optimizes both inference and training processes and significantly minimizes VRAM demands. To assess the impact, we performed experiments using the A100-80GB and obtained the following results. The quantization is implemeted by [bitsandbytes](https://github.com/TimDettmers/bitsandbytes). Check out the [paper](https://arxiv.org/abs/2110.02861) to learn more.

- **BatchSize=1** for fair comparison 

| Model              | Max Length | Task/Dataset                                                                                                            | Precision | Batch Size | Inference | Training             | Single GPU |
|:------------------:|:----------:|:-----------------------------------------------------------------------------------------------------------------------:|:---------:|:----------:|:---------:|:--------------------:|:----------:|
| LLaMA2-70B         | 512        | Language-only/[Alpaca](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json) | BF16      | 1          | 145 GB    | 165 GB (NormBias)    | ❌          |
| LLaMA2-70B         | 512        | Language-only/[Alpaca](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json) | NF4       | 1          | 36 GB     | 46 GB (NormBias)     | ✔          |
| LLaMA2-13B Q-Fomer | 512        | Multi-modal/[LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)             | BF16      | 1          | 31 GB     | 38 GB (NormBiasLoRA) | ✔          |
| LLaMA2-13B Q-Fomer | 512        | Multi-modal [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)             | NF4       | 1          | 13 GB     | 15 GB (NormBiasLoRA) | ✔          |

- **GPU hours** of finetuning

> Note that we use 8x A100-80GB GPU cards for finetuning. The GPU hour refers to `number_of_cards * total_training_time`.

| Model               | Task / Dataset                                                                                                          | Samples | Epoch | Precision | GPU Hours | 8x A100 Training Time |
|:-------------------:|:-----------------------------------------------------------------------------------------------------------------------:|:-------:|:-----:|:---------:|:---------:|:---------------------:|
| LLaMA2-70B          | Language-only/[Alpaca](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json) | 52K     | 4     | BF16      | 100h      | 12.5h                 |
| LLaMA2-70B          | Language-only/[Alpaca](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json) | 52K     | 4     | NF4       | 80h       | 10h                   |
| LLaMA2-13B Q-Former | Multi-modal/[LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)             | 150K    | 3     | BF16      | 170h      | 20h                   |
| LLaMA2-13B Q-Former | Multi-modal/[LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)             | 150K    | 3     | NF4       | 88h       | 11h                   |

## Inference

The trainable weights are saved in `outdir` when QPEFT is done. Run with following scripts:

- Language-only LLaMA2

```bash
# if NormBias
peft_config=""
# elif NormBiasLora
peft_config="configs/model/finetune/sg/llamaPeft_normBiasLora.json"

torchrun --nproc-per-node=1  demos/single_turn.py \
--llama_type "llama_peft"
--llama_config </path/to/params.json> $peft_config \
--tokenizer_path </path/to/tokenizer.model> \
--pretrained_path </path/to/llama>  </path/to/trainable/params> \
--quant
```

- Multi-modal LLaMA2

```bash
# if NormBias
peft_config=""
# elif NormBiasLora
peft_config="configs/model/finetune/sg/llamaPeft_normBiasLora.json"

torchrun --nproc-per-node=1  demos/single_turn_mm.py \
--llama_type "llama_qformerv2_peft"
--llama_config </path/to/params.json> $peft_config \
--tokenizer_path </path/to/tokenizer.model> \
--pretrained_path </path/to/multi/modal/llama>  </path/to/trainable/params> \
--quant
```

Check [inference.md](https://github.com/Alpha-VLLM/LLaMA2-Accessory/blob/quantization/docs/inference.md) for more details.
