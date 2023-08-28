import functools
import math
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_default_linear_weight_init_fn():
    return functools.partial(nn.init.kaiming_uniform_, a=math.sqrt(5))


def get_default_linear_bias_init_fn(fan_in):
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    return functools.partial(nn.init.uniform_, a=-bound, b=bound)


class Linear(nn.Module):
    r"""A Linear module mostly compatible with the PyTorch v2.0.1 builtin one,
    with additional initializer args ``*_init_method``. This is to support
    deferred custom weight initialization: We expect that parameters are
    materialized and set by calling ``Module.reset_parameters()`` in deferred
    initialization, but the ``reset_parameters`` of the builtin ``Linear``
    layer always uses default initialization, making custom initialization
    (e.g., ``xavier_uniform`` or zero initialization) impossible. We
    reimplement a Linear module whose ``reset_parameter`` method respects the
    initializers passed in by the user.

    Args:
        in_features (int): Input feature dimension.
        out_features (int): Output feature dimension.
        bias (bool): Whether a learnable bias is added. Default is ``False``.
        weight_init_fn (Callable[[torch.Tensor], Any], optional): Initializer
            function of the ``weight`` parameter. If not set, follows the
            default initialization of the builtin ``nn.Linear``.
        bias_init_fn (Callable[[torch.Tensor], Any], optional): Initializer
            function of the ``bias`` parameter. If not set, follows the default
            initialization of the builtin ``nn.Linear``.
        device: The device to be passed into the factory function when creating
            the parameter tensors.
        dtype: The dtype to be passed into the factory function when creating
            the parameter tensors.
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True,
        weight_init_fn: Optional[Callable[[torch.Tensor], Any]] = None,
        bias_init_fn: Optional[Callable[[torch.Tensor], Any]] = None,
        device=None, dtype=None,
    ) -> None:
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight_init_fn = weight_init_fn
        self.bias_init_fn = bias_init_fn

        self.weight = nn.Parameter(
            torch.empty([out_features, in_features], **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty([out_features], **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parametes()

    def reset_parametes(self) -> None:
        if not self.weight.is_meta:
            weight_init_fn = (
                self.weight_init_fn or get_default_linear_weight_init_fn()
            )
            weight_init_fn(self.weight.data)
        if self.bias is not None and not self.bias.is_meta:
            bias_init_fn = (
                self.bias_init_fn
                or get_default_linear_bias_init_fn(self.in_features)
            )
            bias_init_fn(self.bias.data)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )
