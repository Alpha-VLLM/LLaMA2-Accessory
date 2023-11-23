import torch
from accessory.model.peft import LoraColumnParallelLinear, LoraRowParallelLinear
from fairscale.nn.model_parallel.layers import ColumnParallelLinear,RowParallelLinear
from accessory.model.meta import MetaModel
from transformers.utils.quantization_config import BitsAndBytesConfig
import bitsandbytes as bnb

from types import MethodType
from tqdm import tqdm

from fairscale.nn.model_parallel.mappings import (
    copy_to_model_parallel_region,
    gather_from_model_parallel_region,
    reduce_from_model_parallel_region,
    scatter_to_model_parallel_region,
)

def forward_ColumnParallelLinear(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
    # Set up backprop all-reduce.
    input_parallel = copy_to_model_parallel_region(input_)
    # Matrix multiply.
    output_parallel = self.quanted_layer(input_parallel)
    if self.bias is not None:
        output_parallel = output_parallel + self.bias
    if self.gather_output:
        # All-gather across the partitions.
        output = gather_from_model_parallel_region(output_parallel)
    else:
        output = output_parallel
    return output

def forward_RowParallelLinear(self, input_: torch.Tensor) -> torch.Tensor:  # type:ignore
    # Set up backprop all-reduce.
    if self.input_is_parallel:
        input_parallel = input_
    else:
        input_parallel = scatter_to_model_parallel_region(input_)
    # Matrix multiply.
    output_parallel = self.quanted_layer(input_parallel)
    # All-reduce across all the partitions.
    output_ = reduce_from_model_parallel_region(output_parallel)
    if self.bias is not None:
        output = output_ + self.bias
    else:
        output = output_
    return output

def forward_LoraColumnParallelLinear(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
    # Set up backprop all-reduce.
    input_parallel = copy_to_model_parallel_region(input_)
    # Matrix multiply.
    output_parallel = self.quanted_layer(input_parallel)
    if self.bias is not None:
        output_parallel = output_parallel + self.bias
    if self.lora_a is not None:
        modification = self.lora_b(self.lora_a(input_))
    else:
        modification = None

    if self.gather_output:
        # All-gather across the partitions.
        output = gather_from_model_parallel_region(output_parallel)
    else:
        output = output_parallel

    if modification is not None:
        output = output + modification
    return output

def forward_LoraRowParallelLinear(self, input_: torch.Tensor) -> torch.Tensor:  # type:ignore
    # Set up backprop all-reduce.
    if self.input_is_parallel:
        input_parallel = input_
    else:
        input_parallel = scatter_to_model_parallel_region(input_)
    # Matrix multiply.
    output_parallel = self.quanted_layer(input_parallel)
    # All-reduce across all the partitions.
    output_ = reduce_from_model_parallel_region(output_parallel)
    if self.lora_a is not None:
        modification = self.lora_b(self.lora_a(input_parallel))
        output_ = output_ + modification
    if self.bias is not None:
        output = output_ + self.bias
    else:
        output = output_
    return output

def forward_Linear(self, input: torch.Tensor) -> torch.Tensor:
    output = self.quanted_layer(input)
    if self.bias != None:
        output += self.bias
    return output

def quantize(
        model : MetaModel,
        quant_conf : BitsAndBytesConfig,
):
    module_list = [_ for _ in model.named_modules() if isinstance(_[1], 
                                                                  (LoraColumnParallelLinear, LoraRowParallelLinear,
                                                                   ColumnParallelLinear, RowParallelLinear, torch.nn.Linear))]
    quant_blocklist = model.get_quant_blocklist()

    for name, module in tqdm(module_list, desc="Qunatization Process"):
        if "lora" in name or name in quant_blocklist:
            continue
        if isinstance(module, (
                               LoraColumnParallelLinear, 
                               LoraRowParallelLinear,
                               ColumnParallelLinear, 
                               RowParallelLinear, 
                               torch.nn.Linear
                               )):
            # 1. Initialize quantization operator
            if quant_conf.load_in_4bit: 
                quanted_layer = bnb.nn.Linear4bit(
                            module.in_features, 
                            module.out_features, 
                            bias=None, 
                            compute_dtype=quant_conf.bnb_4bit_compute_dtype, 
                            compress_statistics=True, 
                            device=None)
                if quant_conf.bnb_4bit_compute_dtype != None:
                    quanted_layer.compute_type_is_set = True

                quanted_layer.weight = bnb.nn.Params4bit(
                            module.weight.data.clone(), 
                            requires_grad=False,
                            quant_type=quant_conf.bnb_4bit_quant_type,
                )

            elif quant_conf.load_in_8bit:
                quanted_layer= bnb.nn.Linear8bitLt(
                            module.in_features, 
                            module.out_features, 
                            bias=None, 
                            has_fp16_weights=quant_conf.llm_int8_has_fp16_weight,
                            threshold=quant_conf.llm_int8_threshold,
                        )
                quanted_layer.weight = bnb.nn.Int8Params(
                            module.weight.data.clone(), 
                            requires_grad=False,
                            #has_fp16_weights=quant_conf.llm_int8_has_fp16_weight,
                        ) 
            else:
                raise NotImplementedError(f'Please determine the proper quantization type.')
            
            # 2. Convert FP layer to quantized layer
            module.quanted_layer = quanted_layer

            if isinstance(module, LoraColumnParallelLinear):
                forward_func = forward_LoraColumnParallelLinear
            elif isinstance(module, LoraRowParallelLinear):
                forward_func = forward_LoraRowParallelLinear
            elif isinstance(module, ColumnParallelLinear):
                forward_func = forward_ColumnParallelLinear
            elif isinstance(module, RowParallelLinear):
                forward_func = forward_RowParallelLinear
            elif isinstance(module, torch.nn.Linear):
                forward_func = forward_Linear
            module.forward = MethodType(forward_func, module)

            del module.weight

