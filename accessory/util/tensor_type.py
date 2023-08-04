from types import TracebackType
from typing import Any, Optional
import torch
import torch.nn as nn


class default_tensor_type:
    _tensor_type_stack = [(torch.float, "cpu")]
    
    def __init__(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
    ) -> None:
        # Only limited combinations are supported.
        assert device is None or device in ["cpu", "cuda"]
        assert dtype is None or dtype in [torch.float, torch.bfloat16, torch.half]
        self.dtype, self.device = dtype, device
    
    def __enter__(self) -> None:
        dtype, device = self.dtype, self.device
        if dtype is None:
            dtype = default_tensor_type._tensor_type_stack[-1][0]
        if device is None:
            device = default_tensor_type._tensor_type_stack[-1][1]
        default_tensor_type._tensor_type_stack.append((dtype, device))
        
        # We use all 3 calls since the new apis (set_default_device, set_default_dtype)
        # seems to be ineffective sometimes (e.g., set_default_device is ineffective to
        # torch.Tensor calls).
        torch.set_default_tensor_type(default_tensor_type.get_tensor_type(dtype, device))
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        default_tensor_type._tensor_type_stack.pop()
        dtype, device = default_tensor_type._tensor_type_stack[-1]

        torch.set_default_tensor_type(default_tensor_type.get_tensor_type(dtype, device))
        torch.set_default_device(device)
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


def promote_trainable_params_to_fp32(model: nn.Module) -> None:
    for param in model.parameters():
        if param.requires_grad:
            if param.is_floating_point() and torch.finfo(param.dtype).bits < 32:
                param.data = param.data.float()
            if param.is_complex() and torch.finfo(param.dtype).bits < 32:
                param.data = param.data.to(torch.complex64)