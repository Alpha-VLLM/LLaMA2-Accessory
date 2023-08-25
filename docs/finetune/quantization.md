# Quantization
## Quantization-Assisted Parameter-Efficient Fine-Tuning
For users with constrained computing resources, we provide an alternative choice through the quantization of the base model, while retaining only carefully selected trainable parameters.
### TL;DR
```bash
# Enable quantization with flag "--quant" and "--only_save_trainable"
torchrun <--some_flags> main_finetune.py <--some_flags> --quant --only_save_trainable
```
For more details, please check {link2repo}`[alpacaLlava_llamaQformerv2Peft_QF_13B](accessory/exps/finetune/mm/alpacaLlava_llamaQformerv2Peft_QF_13B.sh)`.
### Comparison
The LLaMA2-Accessory offers the option to load in 4-bit (NF4), optimizing both inference and training processes while significantly minimizing VRAM demands. To assess its impact, we performed experiments using the A100-80GB and obtained the following results.
#### BatchSize=1
| Model | Max Length | Task/Dataset | Precision | Batch Size | Inference |    Training   |
|:-----:|:----------:|:-------:|:---------:|:----------:|:---------:|:-------------:|
|  LLaMA2-70B  |     512    |  Single-turn Dialogue/[💾Alpaca](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json) |    BF16   |      1     |   145 GB  | 165 GB (PEFT) |
|  LLaMA2-70B  |     512    |  Single-turn Dialogue/[💾Alpaca](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json) |    NF4    |      1     |   36 GB   |  46 GB (PEFT) |
|  LLaMA2-13B+Qfomer  |     512    |  Multi-modal Dialogue/[💾LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main) |    BF16   |      1     |   31 GB  | 38 GB (PEFT) |
|  LLaMA2-13B+Qfomer  |     512    |  Multi-modal Dialogue/[💾LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main) |    NF4    |      1     |   13 GB   |  15 GB (PEFT) |

#### GPU hours of fine-tuning

Note that we use 8x A100-80GB GPU cards for fine-tuning. The GPU hour refers to `number_of_cards * total_training_time`.

|       Model       |               Task / Dataset               | Samples | Epoch | Precision | GPU Hours | 8xA100 Training Time |
|:-----------------:|:------------------------------------------:|:-------:|:-----:|:---------:|:---------:|:--------------------:|
|     LLaMA-70B     |        Single-turn Dialogue / Alpaca       |   52K   |   4   |  BF16     |  100h     |   12.5h              |
|     LLaMA-70B     |        Single-turn Dialogue / Alpaca       |   52K   |   4   |  NF4      |  80h      |   10h                |
| LLaMA-13B+Qformer | Multi-modal Dialogue / LLaVA-Instruct-150K |   150K  |   3   |  BF16     |  170h     |   20h                |
| LLaMA-13B+Qformer | Multi-modal Dialogue / LLaVA-Instruct-150K |   150K  |   3   |  NF4      |  88h      |   11h                |


