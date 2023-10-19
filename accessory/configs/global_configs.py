import warnings

try:
    from flash_attn import flash_attn_func
    USE_FLASH_ATTENTION=True
except ImportError:
    warnings.warn("Cannot import flash_attn, switch to vanilla implementation. ")
    USE_FLASH_ATTENTION=False
