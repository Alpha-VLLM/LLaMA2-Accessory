import functools
import math
import warnings
from typing import (
    Iterable,
    List,
    Union,
)

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed.fsdp._common_utils import (
    TrainingState,
)
from torch.distributed.fsdp._runtime_utils import (
    _lazy_init,
)
from fairscale.nn.model_parallel import initialize as fs_init


def _get_grad_norm(
    params: Iterable[nn.Parameter],
    norm_type: float,
) -> torch.Tensor:
    """
    Returns the gradient norm of parameters ``param`` s, where the gradients
    are viewed as a single vector. The returned norm is in FP32 even if
    parameters/gradients are in a low precision. This is because the downstream
    use of this return value is a reduction across ranks.
    """
    params_with_grad = [param for param in params if param.grad is not None]
    if len(params_with_grad) == 0:
        return torch.tensor(0.0)
    grads = [param.grad for param in params_with_grad]
    grad_dtypes = {grad.dtype for grad in grads}
    if len(grad_dtypes) != 1:
        raise ValueError(
            f"Requires uniform dtype across all gradients but got {grad_dtypes}"
        )
    # Compute the gradient norm in FP32, where we treat the gradients as a
    # single vector
    grad_norm = torch.linalg.vector_norm(
        torch.stack(
            [
                torch.linalg.vector_norm(grad.detach(), norm_type, dtype=torch.float32)
                for grad in grads
            ],
        ),
        norm_type,
        dtype=torch.float32,
    )
    return grad_norm



@torch.no_grad()
def clip_grad_norm(
    model, max_norm: Union[float, int], norm_type: Union[float, int] = 2.0
) -> torch.Tensor:
    """
    Clips the gradient norm of all parameters. The norm is computed over
    all parameters' gradients as viewed as a single vector, and the
    gradients are modified in-place.

    Args:
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'``
            for infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).

    .. note:: If every FSDP instance uses ``NO_SHARD``, meaning that no
        gradients are sharded across ranks, then you may directly use
        :param model:
        :func:`torch.nn.utils.clip_grad_norm_`.

    .. note:: If at least some FSDP instance uses a sharded strategy (i.e.
        one other than ``NO_SHARD``), then you should use this method
        instead of :func:`torch.nn.utils.clip_grad_norm_` since this method
        handles the fact that gradients are sharded across ranks.

    .. note:: The total norm returned will have the "largest" dtype across
        all parameters/gradients as defined by PyTorch's type promotion
        semantics. For example, if *all* parameters/gradients use a low
        precision dtype, then the returned norm's dtype will be that low
        precision dtype, but if there exists at least one parameter/
        gradient using FP32, then the returned norm's dtype will be FP32.

    .. warning:: This needs to be called on all ranks since it uses
        collective communications.
    """

    mp_rank = fs_init.get_model_parallel_rank()
    mp_world_size = fs_init.get_model_parallel_world_size()
    dp_rank = fs_init.get_data_parallel_rank()
    cal_non_split_norm = mp_rank == 0

    _lazy_init(model, model)
    if not model._is_root:
        raise RuntimeError(
            "`clip_grad_norm_()` should only be called on the root FSDP instance"
        )
    model._assert_state(TrainingState.IDLE)

    # Otherwise, there exists some FSDP instance using a sharded strategy,
    # where sharded and non-sharded parameters must be handled separately
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    sharded_params = set()
    nonsharded_params = set()  # `NO_SHARD` or not FSDP-managed
    model_parallel_ignore_params = set()
    model_parallel_params = set()
    grads: List[torch.Tensor] = []
    for handle in traversal_utils._get_fsdp_handles(model):
        target_set = (
            sharded_params if handle.uses_sharded_strategy else nonsharded_params
        )
        if handle._use_orig_params:
            for param in handle.flat_param._params:
                if getattr(param, "is_model_parallel", False) or cal_non_split_norm:
                    target_set.add(param)
                else:
                    model_parallel_ignore_params.add(param)

                if getattr(param, "is_model_parallel", False):
                    model_parallel_params.add(param)
        else:
            raise NotImplementedError("FSD use_orig_params is needed for grad clip with model parallel")
            # target_set.add(handle.flat_param)
            # if handle.flat_param.grad is not None:
            #     grads.append(handle.flat_param.grad)
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad)

        not_fsdp_managed = (
            param not in sharded_params and param not in nonsharded_params and param not in model_parallel_ignore_params
        )
        if not_fsdp_managed:
            if getattr(param, "is_model_parallel", False) or cal_non_split_norm:
                nonsharded_params.add(param)
            else:
                model_parallel_ignore_params.add(param)

            if getattr(param, "is_model_parallel", False):
                model_parallel_params.add(param)
    # Compute local norms (forced to be in FP32)
    local_sharded_norm = _get_grad_norm(sharded_params, norm_type).to(
        model.compute_device
    )
    local_nonsharded_norm = _get_grad_norm(nonsharded_params, norm_type).to(
        model.compute_device
    )
    # Warn if mp_world_size > 1 but model_parallel_params is empty
    if mp_world_size > 1 and len(model_parallel_params) == 0:
        Warning("mp_world_size > 1 but model_parallel_params is empty, are model parallel params correctly marked?")
    # print(f"len of mp_parallel_params: {len(model_parallel_params)}")
    # print(f"[{dp_rank}][{mp_rank}]: sharded [{len(sharded_params)}] unsharded [{len(nonsharded_params)}]", force=True)

    # Reconstruct the total gradient norm depending on the norm type
    if norm_type == math.inf:
        total_norm = torch.maximum(local_sharded_norm, local_nonsharded_norm)
        dist.all_reduce(
            total_norm, op=torch.distributed.ReduceOp.MAX, group=model.process_group
        )
        dist.all_reduce(
            total_norm, op=torch.distributed.ReduceOp.MAX, group=fs_init.get_model_parallel_group()
        )
    else:
        total_norm = local_sharded_norm**norm_type
        dist.all_reduce(total_norm, group=model.process_group)
        # All-reducing the local non-sharded norm would count it an extra
        # world-size-many times
        total_norm += local_nonsharded_norm**norm_type
        # print(f"[{dp_rank}][{mp_rank}]: total1 [{total_norm.item()}]", force=True)
        dist.all_reduce(
            total_norm, group=fs_init.get_model_parallel_group()
        )
        # print(f"[{dp_rank}][{mp_rank}]: total2 [{total_norm.item()}]", force=True)
        total_norm = total_norm ** (1.0 / norm_type)
    if model.cpu_offload.offload_params:
        total_norm = total_norm.cpu()

    clip_coef = max_norm / (total_norm + 1e-6)
    # Multiplying by the clamped coefficient is meaningless when it is
    # equal to 1, but it avoids the host-device sync that would result from
    # `if clip_coef < 1`
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for grad in grads:
        grad.detach().mul_(clip_coef_clamped.to(grad.device, grad.dtype))
    # Use the "largest" dtype by type promotion semantics to use the same
    # dtype as if we did not force local norm computation to be in FP32
    if len(grads) == 0:
        # If this rank has no gradients, then we must default to FP32
        # unless we use additional communication, which we prefer to avoid
        # since `clip_grad_norm_()` is called in the training loop
        warnings.warn(
            f"Called FSDP.clip_grad_norm_() on rank {dist.get_rank()} with no "
            "gradients -- returning the total norm in the default dtype "
            f"{total_norm.dtype}"
        )  # warn since this is generally unexpected
        return total_norm
    total_norm_dtype = functools.reduce(
        lambda dtype1, dtype2: torch.promote_types(dtype1, dtype2),
        [grad.dtype for grad in grads],
    )
    return total_norm.to(total_norm_dtype)