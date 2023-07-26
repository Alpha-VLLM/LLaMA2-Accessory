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
