from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_world_size
)
from fairscale.nn.model_parallel.mappings import (
    copy_to_model_parallel_region,
    gather_from_model_parallel_region,
    scatter_to_model_parallel_region,
    reduce_from_model_parallel_region,
)

from ..linear import (
    get_default_linear_weight_init_fn,
    get_default_linear_bias_init_fn,
)
from .utils import init_tensor_parallel_weights


class ColumnParallelLinear(nn.Module):
    r"""Linear layer with column-wise tensor parallelism. A column parallel
    linear layer expects that the input tensor is replicated among tensor
    parallel ranks, and each rank calculate a part of the output dimensions.

    Args:
        in_features (int): Input feature dimension.
        out_features (int): Output feaature dimension.
        bias (bool): Whether a learnable bias is added. Default is ``False``.
        weight_init_fn (Callable[[torch.Tensor], Any], optional): Initializer
            function of the ``weight`` parameter. If not set, follows the
            default initialization of the builtin ``nn.Linear``. The given
            function should assume that the input tensor is unsharded and the
            distribution of the tensor is taken care of by the outside logic.
        bias_init_fn (Callable[[torch.Tensor], Any], optional): Initializer
            function of the ``bias`` parameter. If not set, follows the default
            initialization of the builtin ``nn.Linear``. The given function
            should assume that the input tensor is unsharded and the
            distribution of the tensor is taken care of by the outside logic.
        gather_output (bool): Whether output should be all-gathered after being
            calculated separately on each rank. Default is ``True``.

    .. note::
        The default initialization of the ``bias`` parameter is different in
        PyTorch and fairscale: The former uses a uniform distribution while the
        latter uses an all-zero constant initialization. We follow the official
        PyTorch behavior. To use the fairscale behavior, pass
        ``torch.nn.init.zeros_`` as the ``bias_init_fn`` argument.
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True,
        weight_init_fn: Optional[Callable[[torch.Tensor], Any]] = None,
        bias_init_fn: Optional[Callable[[torch.Tensor], Any]] = None,
        gather_output: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_init_fn = weight_init_fn
        self.bias_init_fn = bias_init_fn
        self.gather_output = gather_output

        tp_world_size = get_model_parallel_world_size()
        assert self.out_features % tp_world_size == 0, (
            "ColumnParallelLinear currently requires that the output "
            "dimension is evenly divisible by the tensor parallel world size."
        )
        self.local_out_features = self.out_features // tp_world_size

        self.weight = nn.Parameter(
            torch.empty([self.local_out_features, in_features])
        )
        if bias:
            self.bias = nn.Parameter(torch.empty([self.local_out_features]))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        weight_init_fn = (
            self.weight_init_fn or get_default_linear_weight_init_fn()
        )
        init_tensor_parallel_weights(self.weight, weight_init_fn, 0)
        if self.bias is not None:
            bias_init_fn = (
                self.bias_init_fn
                or get_default_linear_bias_init_fn(self.in_features)
            )
            init_tensor_parallel_weights(self.bias, bias_init_fn, 0)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        input_parallel = copy_to_model_parallel_region(input_)
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        if self.gather_output:
            output = gather_from_model_parallel_region(output_parallel)
            return output
        else:
            return output_parallel

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"local_out_features={self.local_out_features}, "
            f"bias={self.bias is not None}, "
            f"gather_output={self.gather_output}"
        )


class RowParallelLinear(nn.Module):
    r"""Linear layer with row-wise tensor parallelism. A row parallel linear
    layer divides the input feature dimensions among the tensor parallel ranks,
    calculates the linear mapping on each part of the dimensions and sum the
    results to form the output.

    Args:
        in_features (int): Input feature dimension.
        out_features (int): Output feaature dimension.
        bias (bool): Whether a learnable bias is added. Default is ``False``.
        weight_init_fn (Callable[[torch.Tensor], Any], optional): Initializer
            function of the ``weight`` parameter. If not set, follows the
            default initialization of the builtin ``nn.Linear``. The given
            function should assume that the input tensor is unsharded and the
            distribution of the tensor is taken care of by the outside logic.
        bias_init_fn (Callable[[torch.Tensor], Any], optional): Initializer
            function of the ``bias`` parameter. If not set, follows the default
            initialization of the builtin ``nn.Linear``. The given function
            should assume that the input tensor is unsharded and the
            distribution of the tensor is taken care of by the outside logic.
        input_is_parallel (bool): If true, assumes that the input tensor is
            already sharded (e.g., the output of a ColumnParallelLinear in
            which ``gather_output=False``).

    .. note::
        The default initialization of the ``bias`` parameter is different in
        PyTorch and fairscale: The former uses a uniform distribution while the
        latter uses an all-zero constant initialization. We follow the official
        PyTorch behavior. To use the fairscale behavior, pass
        ``torch.nn.init.zeros_`` as the ``bias_init_fn`` argument.
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True,
        weight_init_fn: Optional[Callable[[torch.Tensor], Any]] = None,
        bias_init_fn: Optional[Callable[[torch.Tensor], Any]] = None,
        input_is_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_init_fn = weight_init_fn
        self.bias_init_fn = bias_init_fn
        self.input_is_parallel = input_is_parallel

        tp_world_size = get_model_parallel_world_size()
        assert self.in_features % tp_world_size == 0, (
            "RowParallelLinear currently requires that the output dimension"
            "is evenly divisible by the tensor parallel world size."
        )
        self.local_in_features = in_features

        self.weight = nn.Parameter(
            torch.empty([self.out_features, self.local_in_features])
        )
        if bias:
            self.bias = nn.Parameter(torch.empty([self.out_features]))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        weight_init_fn = (
            self.weight_init_fn or get_default_linear_weight_init_fn()
        )
        init_tensor_parallel_weights(self.weight, weight_init_fn, 1)
        if self.bias is not None:
            bias_init_fn = (
                self.bias_init_fn
                or get_default_linear_bias_init_fn(self.in_features)
            )
            init_tensor_parallel_weights(self.bias, bias_init_fn, -1)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        input_parallel = (
            input_ if self.input_is_parallel else
            scatter_to_model_parallel_region(input_)
        )
        output_parallel = F.linear(input_parallel, self.weight)
        output = reduce_from_model_parallel_region(output_parallel)
        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"local_in_features={self.local_in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"input_is_parallel={self.input_is_parallel}"
        )
