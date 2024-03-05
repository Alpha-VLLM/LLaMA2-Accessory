from typing import Dict
import torch
import torch.nn as nn
import torch.distributed as dist
import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear, RowParallelLinear, ParallelEmbedding
)


def get_model_parallel_dim_dict(model: nn.Module) -> Dict[str, int]:
    ret_dict = {}
    for module_name, module in model.named_modules():
        def param_fqn(param_name):
            return param_name if module_name == "" else module_name + "." + param_name
        if isinstance(module, ColumnParallelLinear):
            ret_dict[param_fqn("weight")] = 0
            if module.bias is not None:
                ret_dict[param_fqn("bias")] = 0
        elif isinstance(module, RowParallelLinear):
            ret_dict[param_fqn("weight")] = 1
            if module.bias is not None:
                ret_dict[param_fqn("bias")] = -1
        elif isinstance(module, ParallelEmbedding):
            ret_dict[param_fqn("weight")] = 1
        else:
            for param_name, param in module.named_parameters(recurse=False):
                ret_dict[param_fqn(param_name)] = -1
    return ret_dict


def calculate_l2_grad_norm(
    model: nn.Module, model_parallel_dim_dict: Dict[str, int],
) -> float:
    mp_norm_sq = torch.tensor(0., dtype=torch.float32, device="cuda")
    non_mp_norm_sq = torch.tensor(0., dtype=torch.float32, device="cuda")

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        name = ".".join(x for x in name.split(".") if not x.startswith("_"))
        assert name in model_parallel_dim_dict
        if model_parallel_dim_dict[name] < 0:
            non_mp_norm_sq += param.grad.norm(dtype=torch.float32) ** 2
        else:
            mp_norm_sq += param.grad.norm(dtype=torch.float32) ** 2

    dist.all_reduce(mp_norm_sq)
    dist.all_reduce(non_mp_norm_sq)
    non_mp_norm_sq /= fs_init.get_model_parallel_world_size()

    return (mp_norm_sq.item() + non_mp_norm_sq.item()) ** 0.5


def scale_grad(model: nn.Module, factor: float) -> None:
    for param in model.parameters():
        if param.grad is not None:
            param.grad.mul_(factor)
