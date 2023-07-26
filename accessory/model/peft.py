import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_

from fairscale.nn.model_parallel.layers import (
    RowParallelLinear,
    ColumnParallelLinear,
)


class LoraColumnParallelLinear(ColumnParallelLinear):
    """ColumnParallelLinear extended with LoRA support"""

    def __init__(self, *args, **kwargs) -> None:
        self.lora_rank = kwargs.pop("lora_rank", 0)
        super().__init__(*args, **kwargs)

        if self.lora_rank > 0:
            self.lora_a = nn.Linear(self.in_features, self.lora_rank, bias=False)
            trunc_normal_(self.lora_a.weight, std=.02)
            self.lora_b = ColumnParallelLinear(
                self.lora_rank, self.out_features, bias=False,
                gather_output=self.gather_output,
            )
            nn.init.zeros_(self.lora_b.weight)
        else:
            self.lora_a = None
            self.lora_b = None

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
        output = super().forward(input_)
        if self.lora_rank > 0:
            output += self.lora_b(self.lora_a(input_))
        return output

    @staticmethod
    def from_non_lora(layer: ColumnParallelLinear, **kwargs) -> LoraColumnParallelLinear:
        new_layer_kwargs = dict(
            in_features=layer.in_features,
            out_features=layer.out_features,
            bias=layer.bias is not None,
            gather_output=layer.gather_output,
            init_method=lambda x: x,
            keep_master_weight_for_test=layer.master_weight is not None,
        )
        new_layer_kwargs.update(kwargs)
        layer_with_lora = LoraColumnParallelLinear(**new_layer_kwargs)
        layer_with_lora.weight.data.copy_(layer.weight)
        if layer_with_lora.bias is not None:
            layer_with_lora.bias.data.copy_(layer.weight)
        return layer_with_lora

class LoraRowParallelLinear(RowParallelLinear):
    """RowParallelLinear with LoRA support"""

    def __init__(self, *args, **kwargs) -> None:
        self.lora_rank = kwargs.pop("lora_rank", 0)
        super().__init__(self, *args, **kwargs)

        if self.lora_rank > 0:
            self.lora_a = RowParallelLinear(
                self.in_features, self.lora_rank, bias=False,
                input_is_parallel=self.input_is_parallel,
            )
            trunc_normal_(self.lora_a.weight, std=.02)
            self.lora_b = nn.Linear(self.lora_rank, self.out_features, bias=False)
            nn.init.zeros_(self.lora_b.weight)
        else:
            self.lora_a = None
            self.lora_b = None

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type:ignore
        output = super().forward(input_)
        if self.lora_rank > 0:
            output += self.lora_b(self.lora_a(input_))
        return output

    @staticmethod
    def from_non_lora(layer: RowParallelLinear, **kwargs) -> LoraRowParallelLinear:
        new_layer_kwargs = dict(
            in_features=layer.in_features,
            out_features=layer.out_features,
            bias=layer.bias is not None,
            input_is_parallel=layer.input_is_parallel,
            init_method=lambda x: x,
            keep_master_weight_for_test=layer.master_weight is not None,
        )
        new_layer_kwargs.update(kwargs)
        layer_with_lora = LoraRowParallelLinear(**new_layer_kwargs)
        layer_with_lora.weight.data.copy_(layer.weight)
        if layer_with_lora.bias is not None:
            layer_with_lora.bias.data.copy_(layer.weight)
        return layer_with_lora

def wrap_lora(layer: nn.Module, **kwargs):
    base_module_to_lora_module = [
        (ColumnParallelLinear, LoraColumnParallelLinear),
        (RowParallelLinear, LoraRowParallelLinear),
    ]
    for base_module, lora_module in base_module_to_lora_module:
        if isinstance(layer, base_module):
            return lora_module.from_non_lora(layer, **kwargs)
    raise NotImplementedError(f"LoRA wrapping for layer of type {type(layer)} is not implemented.")
