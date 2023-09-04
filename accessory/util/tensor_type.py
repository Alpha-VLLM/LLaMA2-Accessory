from types import TracebackType
from typing import Any, List, Optional, Tuple
import torch
import torch.nn as nn


class default_tensor_type:
    r"""A context manager that maintains a stack of tensor type states. Each
    state is a tuple of 3 elements: (1) The default scalar dtype of new
    tensors; (2) The default real device of the new tensors (i.e., not
    including the ``meta`` device) and (3) Whether new tensors should be
    created as ``meta``.

    Each argument is optional and will inherit the last value on the stack if
    passed ``None``.

    .. note::
        Unlike PyTorch which manages ``meta`` as a special type of device, we
        manage ``is_meta`` as a separate dimension in our states. This allows
        us to maintain the materialization device while entering or exiting
        ``meta`` creation state freely.

    Args:
        dtype (torch.dtype, Optional): The scalar data type of the new tensors.
        device (str, Optional): The string representing the real device of the
            new tensors. ``meta`` device and device ordinals are not supported.
        is_meta (bool, Optional): Whether new tensors should be created as
            ``meta``.
    """

    _tensor_type_stack: List[Tuple[torch.dtype, str, bool]] = [
        (torch.float, "cpu", False)
    ]

    def __init__(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
        is_meta: Optional[bool] = None,
    ) -> None:
        # Only limited combinations are supported.
        assert device is None or device in ["cpu", "cuda"]
        assert dtype is None or dtype in [torch.float, torch.bfloat16,
                                          torch.half]
        self.dtype, self.device, self.is_meta = dtype, device, is_meta

    def __enter__(self) -> None:
        dtype, device, is_meta = self.dtype, self.device, self.is_meta
        if dtype is None:
            dtype = default_tensor_type._tensor_type_stack[-1][0]
        if device is None:
            device = default_tensor_type._tensor_type_stack[-1][1]
        if is_meta is None:
            is_meta = default_tensor_type._tensor_type_stack[-1][2]
        default_tensor_type._tensor_type_stack.append((dtype, device, is_meta))
        default_tensor_type._set_pytorch_state_by_last_state_tuple()

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        default_tensor_type._tensor_type_stack.pop()
        default_tensor_type._set_pytorch_state_by_last_state_tuple()

    @staticmethod
    def _set_pytorch_state_by_last_state_tuple():
        device, dtype, is_meta = default_tensor_type._tensor_type_stack[-1]

        # We use all 3 calls since the new apis (set_default_device,
        # set_default_dtype) seems to be ineffective sometimes (e.g.,
        # set_default_device is ineffective to torch.Tensor calls).
        #
        # We are aware that torch.Tensor creator is deprecated as of PyTorch
        # v2.0.1. This is a 'catch-all' for some third-party libraries (e.g.,
        # fairscale) which still uses the old torch.Tensor API but is out of
        # our control.
        #
        # Also, torch.set_default_tensor_type seems to not support the new
        # meta tensor feature so we have to fall back to the real device.
        torch.set_default_tensor_type(
            default_tensor_type.get_tensor_type(dtype, device)
        )
        torch.set_default_device(device if not is_meta else "meta")
        torch.set_default_dtype(dtype)

    @staticmethod
    def get_tensor_type(dtype: torch.dtype, device: str) -> Any:
        return {
            (torch.float, "cpu"): torch.FloatTensor,
            (torch.bfloat16, "cpu"): torch.BFloat16Tensor,
            (torch.half, "cpu"): torch.HalfTensor,
            (torch.float, "cuda"): torch.cuda.FloatTensor,
            (torch.bfloat16, "cuda"): torch.cuda.BFloat16Tensor,
            (torch.half, "cuda"): torch.cuda.HalfTensor,
        }[(dtype, device)]

    @staticmethod
    def get_current_materialization_device() -> torch.device:
        r"""Get the current 'real' device on the default tensor type stack,
        regardless of the is_meta state.
        """
        return torch.device(default_tensor_type._tensor_type_stack[-1][1])


def promote_trainable_params_to_fp32(model: nn.Module) -> None:
    r"""This method promotes each parameter of a given model with
    ``requires_grad=True`` to at least FP32, following the common practice of
    mixed precision training that a copy of FP32 master weights is maintained
    for optimization despite that each forward and backward pass uses the
    down-casted low precision weights (16-bit, or even 8-bit on the newer
    hardware).

    .. note::
        The method handles both floating point (real) and complex scalar types.
        For complex type, both the real and the imaginary parts are promoted to
        FP32 (resulting in the ``torch.complex64`` scalar type).

    Args:
        model (torch.nn.Module): The model whose ``requires_grad`` parameters
            are promoted to FP32.
    """
    for param in model.parameters():
        if param.requires_grad:
            if param.is_floating_point():
                if torch.finfo(param.dtype).bits < 32:
                    param.data = param.data.float()
            elif param.is_complex():
                if torch.finfo(param.dtype).bits < 32:
                    param.data = param.data.to(torch.complex64)
