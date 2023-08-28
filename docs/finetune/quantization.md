# Fine-Tuning with Quantization

For users with constrained computing resources, we provide an alternative choice through the quantization of the base model, while retaining only carefully selected trainable parameters.

We support Quantized PEFT methods including QLoRA, QNormBias and QNormBiasLoRA, which significantly minimize the computing demands. 

### Best Practice

```bash
# Enable quantization with flag "--quant" and "--only_save_trainable"
torchrun <--some_flags> main_finetune.py <--some_flags> --quant --only_save_trainable
```

For more details, please check the following links:

| Method        | Fine-tune?Language-only LLaMA 2                                                                                                                                | Fine-tune?Multi-Modal LLaMA2                                                                                                                                             |
|:-------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| QLoRA         | -                                                                                                                                                              | -                                                                                                                                                                        |
| QNormBias     | [alpaca_llamaPeft_normBias_QF.sh](https://github.com/Alpha-VLLM/LLaMA2-Accessory/blob/main/accessory/exps/finetune/sg/alpaca_llamaPeft_normBias_QF.sh)         | -                                                                                                                                                                        |
| QNormBiasLoRA | [alpaca_llamaPeft_normBiasLora_QF.sh](https://github.com/Alpha-VLLM/LLaMA2-Accessory/blob/main/accessory/exps/finetune/sg/alpaca_llamaPeft_normBiasLora_QF.sh) | [alpacaLlava_llamaQformerv2Peft_QF_13B.sh](https://github.com/Alpha-VLLM/LLaMA2-Accessory/blob/main/accessory/exps/finetune/mm/alpacaLlava_llamaQformerv2Peft_QF_13B.sh) |



### Comparison

We offer the option to load in 4/8 bit of quantization setting, optimizing both inference and training processes while significantly minimizing VRAM demands. To assess its impact, we performed experiments using the A100-80GB and obtained the following results.

#### BatchSize=1

| Model             | Max Length | Task/Dataset                                                                                                            | Precision | Batch Size | Inference | Training          |
|:-----------------:|:----------:|:-----------------------------------------------------------------------------------------------------------------------:|:---------:|:----------:|:---------:|:-----------------:|
| LLaMA2-70B        | 512        | Language-only/[Alpaca](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json) | BF16      | 1          | 145 GB    | 165 GB (NormBias) |
| LLaMA2-70B        | 512        | Language-only/[Alpaca](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json) | NF4       | 1          | 36 GB     | 46 GB (NormBias)  |
| LLaMA2-13B+Qfomer | 512        | Multi-modal/[LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)             | BF16      | 1          | 31 GB     | 38 GB (NormBias)  |
| LLaMA2-13B+Qfomer | 512        | Multi-modal [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)             | NF4       | 1          | 13 GB     | 15 GB (NormBias)  |

#### GPU hours of fine-tuning

Note that we use 8x A100-80GB GPU cards for fine-tuning. The GPU hour refers to `number_of_cards * total_training_time`.

| Model             | Task / Dataset                                                                                                          | Samples | Epoch | Precision | GPU Hours | 8xA100 Training Time |
|:-----------------:|:-----------------------------------------------------------------------------------------------------------------------:|:-------:|:-----:|:---------:|:---------:|:--------------------:|
| LLaMA-70B         | Language-only/[Alpaca](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json) | 52K     | 4     | BF16      | 100h      | 12.5h                |
| LLaMA-70B         | Language-only/[Alpaca](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json) | 52K     | 4     | NF4       | 80h       | 10h                  |
| LLaMA-13B+Qformer | Multi-modal/[LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)             | 150K    | 3     | BF16      | 170h      | 20h                  |
| LLaMA-13B+Qformer | Multi-modal/[LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)             | 150K    | 3     | NF4       | 88h       | 11h                  |
