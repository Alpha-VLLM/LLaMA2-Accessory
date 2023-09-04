import functools

import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_

from .layers import ColumnParallelLinear, RowParallelLinear, Linear


class LoraColumnParallelLinear(ColumnParallelLinear):
    r"""ColumnParallelLinear with LoRA. For unlisted arguments see the
    documentation for ``ColumnParallelLinear``.

    Args:
        lora_rank (int): Bottleneck dimension in the LoRA projections. Default
            to ``0``. Only supported as kwargs.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.lora_rank = kwargs.pop("lora_rank", 0)
        super().__init__(*args, **kwargs)

        if self.lora_rank > 0:
            self.lora_a = Linear(
                self.in_features, self.lora_rank, bias=False,
                weight_init_fn=functools.partial(trunc_normal_, std=.02)
            )
            self.lora_b = ColumnParallelLinear(
                self.lora_rank, self.out_features, bias=False,
                weight_init_fn=nn.init.zeros_,
                gather_output=self.gather_output,
            )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        output = super().forward(input_)
        if self.lora_rank > 0:
            output = output + self.lora_b(self.lora_a(input_))
        return output


class LoraRowParallelLinear(RowParallelLinear):
    r"""RowParallelLinear with LoRA. For unlisted arguments see the
    documentation for ``RowParallelLinear``.

    Args:
        lora_rank (int): Bottleneck dimension in the LoRA projections. Default
            to ``0``. Only supported as kwargs.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.lora_rank = kwargs.pop("lora_rank", 0)
        super().__init__(*args, **kwargs)

        if self.lora_rank > 0:
            self.lora_a = RowParallelLinear(
                self.in_features, self.lora_rank, bias=False,
                weight_init_fn=functools.partial(trunc_normal_, std=.02),
                input_is_parallel=self.input_is_parallel,
            )
            self.lora_b = Linear(
                self.lora_rank, self.out_features, bias=False,
                weight_init_fn=nn.init.zeros_,
            )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        output = super().forward(input_)
        if self.lora_rank > 0:
            output = output + self.lora_b(self.lora_a(input_))
        return output
