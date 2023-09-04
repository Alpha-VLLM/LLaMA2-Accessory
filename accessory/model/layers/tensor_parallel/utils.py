from typing import Any, Callable

import torch
import torch.distributed as dist

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_group,
    get_model_parallel_src_rank,
    get_model_parallel_rank,
    get_model_parallel_world_size,
)


def _broadcast_replicated_tensor(tensor: torch.Tensor) -> None:
    group = get_model_parallel_group()
    backend = dist.get_backend(group)
    reduction_device = "cuda" if backend == "nccl" else tensor.device

    bcast_tensor = tensor.to(reduction_device)
    dist.broadcast(bcast_tensor, get_model_parallel_src_rank(), group)
    if bcast_tensor is not tensor:
        tensor.copy_(bcast_tensor)


def _scatter_distributed_tensor(
    tensor: torch.Tensor, master_tensor: torch.Tensor, shard_dim: int
) -> None:
    group = get_model_parallel_group()
    backend = dist.get_backend(group)
    reduction_device = "cuda" if backend == "nccl" else tensor.device

    if get_model_parallel_rank() == 0:
        master_tensor = master_tensor.to(reduction_device)
        recv_tensor = tensor.to(reduction_device)
        dist.scatter(recv_tensor, master_tensor.split(tensor.size(shard_dim)),
                     get_model_parallel_src_rank(), group)
    else:
        recv_tensor = tensor.to(reduction_device)
        dist.scatter(recv_tensor, None, get_model_parallel_src_rank(), group)
    if recv_tensor is not tensor:
        tensor.copy_(recv_tensor)


def init_tensor_parallel_weights(
    tensor: torch.Tensor, init_fn: Callable[[torch.Tensor], Any],
    shard_dim: int = -1
) -> None:
    r"""This is a helper function that initializes a tensor-parallel tensor
    from a regular tensor-parallel-unaware ``init_fn``. A typical use case is
    that ``init_fn`` may calculate the initialization statistics based on the
    ``fan_in`` or ``fan_out`` measured with the shape of the tensor which
    will be incorrect if the tensor is sharded across tensor-parallel ranks.
    Thus, we create a helper function that initializes a tensor as a whole and
    then distribute it across the model parallel ranks.

    Args:
        tensor (torch.Tensor): The (tensor-parallel-sharded) tensor to
            initialize.
        init_fn (Callable[[torch.Tensor], Any]): The tensor-parallel-unaware
            initializer to be called on the unsharded weights.
        shard_dim (int): The sharding dimension of the tensor. If < 0, the
            tensor is treated as replicated. Default is -1.
    """
    if tensor.is_meta:
        return

    if shard_dim < 0:
        if get_model_parallel_rank() == 0:
            init_fn(tensor.data)
        _broadcast_replicated_tensor(tensor.data)
        return

    if get_model_parallel_rank() == 0:
        master_tensor_shape = list(tensor.size())
        master_tensor_shape[shard_dim] *= get_model_parallel_world_size()
        master_tensor = torch.empty(master_tensor_shape,
                                    device=tensor.device, dtype=tensor.dtype)
    else:
        master_tensor = None
    init_fn(master_tensor)
    _scatter_distributed_tensor(tensor.data, master_tensor, shard_dim)
