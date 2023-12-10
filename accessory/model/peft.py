from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Parameter, init

from fairscale.nn.model_parallel.initialize import get_model_parallel_rank, get_model_parallel_world_size
from fairscale.nn.model_parallel.mappings import (
    copy_to_model_parallel_region,
    gather_from_model_parallel_region,
    reduce_from_model_parallel_region,
    scatter_to_model_parallel_region,
)
from fairscale.nn.model_parallel.utils import VocabUtility, divide_and_check_no_remainder
from fairscale.nn.model_parallel.layers import _initialize_affine_weight
from fairscale.nn.model_parallel.layers import (
    RowParallelLinear,
    ColumnParallelLinear,
)


class LoraLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank = 0
    ):
        super().__init__(in_features, out_features, bias)

        self.lora_rank = lora_rank
        if self.lora_rank > 0:
            self.lora_a = nn.Linear(self.in_features, self.lora_rank, bias=False)
            # workaround because trunc_normal_ does not currently support bfloat16
            _ = init.trunc_normal_(self.lora_a.weight.data.to(torch.float32), std=.02)
            self.lora_a.weight.data.copy_(_)
            self.lora_b = nn.Linear(self.lora_rank, self.out_features, bias=False)
            nn.init.zeros_(self.lora_b.weight)
        else:
            self.lora_a = None
            self.lora_b = None

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type:ignore
        # Matrix multiply.
        output = F.linear(input_, self.weight, self.bias)
        if self.lora_a is not None:
            modification = self.lora_b(self.lora_a(input_))
        else:
            modification = None

        if modification is not None:
            output = output + modification
        return output


class LoraColumnParallelLinear(ColumnParallelLinear):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        lora_rank=0
    ) -> None:
        nn.Module.__init__(self)

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.output_size_per_partition = divide_and_check_no_remainder(out_features, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(torch.Tensor(self.output_size_per_partition, self.in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size_per_partition))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        # Initialize weight.
        self.master_weight = _initialize_affine_weight(
            self.weight,
            self.out_features,
            self.in_features,
            self.output_size_per_partition,
            0,
            init_method,
            stride=stride,
            return_master_weight=keep_master_weight_for_test,
        )
        
        self.lora_rank = lora_rank
        if self.lora_rank > 0:
            # if world_size > 1:
            #     raise NotImplemented("Lora with model parallel with change the original behavior, not yet supported")
            self.lora_a = nn.Linear(self.in_features, self.lora_rank, bias=False)
            # workaround because trunc_normal_ does not currently support bfloat16
            _ = init.trunc_normal_(self.lora_a.weight.data.to(torch.float32), std=.02)
            self.lora_a.weight.data.copy_(_)
            self.lora_b = ColumnParallelLinear(self.lora_rank, self.out_features, bias=False, gather_output=gather_output)
            nn.init.zeros_(self.lora_b.weight)
        else:
            self.lora_a = None
            self.lora_b = None

    def get_master_weight(self) -> torch.Tensor:
        return gather_from_model_parallel_region(self.weight.data.transpose(0, 1)).transpose_(0, 1)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Set up backprop all-reduce.
        input_parallel = copy_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
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


class LoraRowParallelLinear(RowParallelLinear):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        lora_rank = 0
    ):
        nn.Module.__init__(self)

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.input_size_per_partition = divide_and_check_no_remainder(in_features, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(torch.Tensor(self.out_features, self.input_size_per_partition))
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        # Initialize weight.
        self.master_weight = _initialize_affine_weight(
            self.weight,
            self.out_features,
            self.in_features,
            self.input_size_per_partition,
            1,
            init_method,
            stride=stride,
            return_master_weight=keep_master_weight_for_test,
        )

        self.lora_rank = lora_rank
        if self.lora_rank > 0:
            # if world_size > 1:
            #     raise NotImplemented("Lora with model parallel with change the original behavior, not yet supported")
            self.lora_a = RowParallelLinear(self.in_features, self.lora_rank, bias=False, input_is_parallel=True)
            # workaround because trunc_normal_ does not currently support bfloat16
            _ = init.trunc_normal_(self.lora_a.weight.data.to(torch.float32), std=.02)
            self.lora_a.weight.data.copy_(_)
            self.lora_b = nn.Linear(self.lora_rank, self.out_features, bias=False)
            nn.init.zeros_(self.lora_b.weight)
        else:
            self.lora_a = None
            self.lora_b = None

    def get_master_weight(self) -> torch.Tensor:
        return gather_from_model_parallel_region(self.weight.data)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type:ignore
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight)
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