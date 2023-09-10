r"""This module includes utility methods to set-up parameter groups.

Some common use cases include: With / Without weight decay for certain types of
parameters; layer-wise learning-rate decay for training visual encoders.
"""

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn


BlockwiseParamGroupFuncType = Callable[[Dict[str, torch.Tensor]],
                                       List[List[str]]]
_LAYERWISE_PARAM_GROUP_FUNCS: Dict[str, BlockwiseParamGroupFuncType] = {}


def _layerwise_param_group_func(
    prefix: str
) -> Callable[[BlockwiseParamGroupFuncType], BlockwiseParamGroupFuncType]:
    r"""Decorator to define a method that generate the block-wise parameter
    group for all keys starting with a specific prefix.
    """
    def inner_func(
        func: BlockwiseParamGroupFuncType
    ) -> BlockwiseParamGroupFuncType:
        global _LAYERWISE_PARAM_GROUP_FUNCS
        assert prefix not in _LAYERWISE_PARAM_GROUP_FUNCS, (
            "Repeated registration of prefix: " + prefix
        )
        _LAYERWISE_PARAM_GROUP_FUNCS[prefix] = func
        return func
    return inner_func


@_layerwise_param_group_func("")
def _make_default_param_group(
    meta_param_dict: Dict[str, torch.Tensor]
) -> List[List[str]]:
    r"""Generate the default param group. As we group parameters according
    to longest name prefix match, this function defines the grouping of the
    empty prefix to act as the catch-all grouping.
    """
    return [list(meta_param_dict.keys())]


@_layerwise_param_group_func("llma.clip.visual.")
def _clip_make_layerwise_param_groups(
    meta_param_dict: Dict[str, torch.Tensor]
) -> List[List[str]]:
    r"""Generate a list of param groups for the clip visual encoder.

    .. note:
        This also serves as a reference for implementing other vision encoder
        grouping in the future. All grouping methods should comply with this
        spec.

    Args:
        meta_param_dict (Dict[str, torch.Tensor]): The param dict received from
            the caller. The values are meta tensors which will reflect the
            shape, dtype and ``requires_grad`` status of the underlying real
            parameter but no actual values are contained. This is to save
            memory when the real parameters may be sharded.

    Returns:
        List[List[str]]: A list of lists. Each inner list represents a
            parameter group sharing the same learning rate. Each inner list
            should contain names of the parameters that belong to this group.
            The groups should be sorted by depth in ascending order (i.e.,
            groups with larger index are closer to the output).
    """
    num_layers = 0
    while f"transformer.resblocks.{num_layers}.ln_1.weight" in meta_param_dict:
        num_layers += 1
    groups = [
        ["class_embedding", "positional_embedding", "conv1.weight",
         "ln_pre.weight", "ln_pre.bias"]
    ]
    for i in range(num_layers):
        layer_prefix = f"transformer.resblocks.{i}."
        layer_params = [
            "ln_1.weight", "ln_1.bias",
            "attn.in_proj_weight", "attn.in_proj_bias",
            "attn.out_proj.weight", "attn.out_proj.bias",
            "ln_2.weight", "ln_2.bias",
            "mlp.c_fc.weight", "mlp.c_fc.bias",
            "mlp.c_proj.weight", "mlp.c_proj.bias",
        ]
        groups.append([layer_prefix + x for x in layer_params])
    groups.append(["ln_post.weight", "ln_post.bias"])

    return groups


# Follow the example of _clip_make_layerwise_param_groups to implement your own
# grouping for new visual backbones.


def make_param_groups(
    model: nn.Module,
    base_lr: float,
    base_weight_decay: float,
    bias_and_1d_params_no_decay: bool = True,
    no_weight_decay_list: List[str] = [],
    layer_wise_lr_decay: Optional[float | Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    r"""This method sets up param groups of different learning rate or weight
    decay configurations. Currently, the supported functions are:

        * Disable weight decay by the default criterion: parameters with names
          ending with ``.bias`` and parameters with dimensions <= 1 (following
          ``timm``, controlled by argument ``bias_and_1d_params_no_decay``).
        * Disable weight decay for parameters whose name is in the argument
          ``no_weight_decay_list``.
        * Layer-wise learning rate decay for vision backbones.

    This method supports FSDP- and checkpointing-wrapped modules.

    Args:
        model (nn.Module): The module whose parameters are grouped.
        base_lr (float): Base learning rate before applying any factors.
        base_weight_decay (float): Base weight decay before applying any
            factors.
        bias_and_1d_params_no_decay (bool): Apply the default heuristic to
            disable weight decay for selected parameters (with names ending
            with ``.bias`` or with shape dimension <= 1, following ``timm``).
            Default to ``True``.
        no_weight_decay_list (List[str]): Parameters with a name in the list
            will have its weight decay disabled. The names in the list should
            be the plain name before FSDP wrapping.
        layer_wise_lr_decay (float | Dict[str, Float], optional): A layer-wise
            learning rate decay factor to apply to vision weights. Supported
            values are: (1) ``None`` for no layer-wise LR decay, (2) a
            ``float`` value for a uniform decay for any vision backbone, and
            (3) a ``dict`` of ``float`` values for applying different decay
            factor for different vision encoders. Default to ``None``.
    """

    # Most of the grouping logic relies on parameter names at this moment.
    # Create the dictionaries with clean names (i.e., FSDP-specific sections
    # removed) for easier implementation.
    clean_name_to_real_param_dict: Dict[str, nn.Parameter] = {}
    # Some grouping logic depend on the parameter shape which may be lost after
    # FSDP wrapping so create meta tensors to reflect the original shape.
    clean_name_to_meta_param_dict: Dict[str, torch.Tensor] = {}

    # FSDP (and checkpoint wrappers) caused the complexity: We have to traverse
    # to find out the clean names and real shapes of the parameters.
    #
    # TODO: The traverse procedure currently rely on some private methods which
    # may change in future PyTorch releases. Be careful when bumping the
    # PyTorch version or write a more robust one in the future.
    from torch.distributed._composable_state import _get_module_state
    from torch.distributed.fsdp._common_utils import _FSDPState, FSDP_PREFIX
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointWrapper, _CHECKPOINT_PREFIX
    )

    def get_fqn(module_name: str, param_name: str):
        return module_name + "." + param_name if module_name else param_name

    def match_and_strip_prefix(prefix: str,
                               src_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key[len(prefix):]: value for key, value in src_dict.items()
            if key.startswith(prefix)
        }

    def dfs_find_params_and_clean_names(
        module: nn.Module, prefix: str,
        sharded_views_original_shape: Dict[str, torch.Size],
    ) -> None:
        state = _get_module_state(module)
        if isinstance(module, CheckpointWrapper):
            # The special points for CheckpointWrappers are:
            # 1. We want to strip the _CHECKPOINT_PREFIX.
            # 2. There might be two cases when CheckpointWrapper is used
            #    with FSPD: FSDP(checkpoint(model)) and
            #    checkpoint(FSDP(model)). As of v2.0.1 it seems to be unclear
            #    which one is the preferred / correct one so handle both for
            #    now.
            new_sharded_views_dict = {}
            for key, value in sharded_views_original_shape.items():
                if key.startswith(_CHECKPOINT_PREFIX):
                    key = key[len(_CHECKPOINT_PREFIX):]
                new_sharded_views_dict[key] = value
            dfs_find_params_and_clean_names(
                module._checkpoint_wrapped_module,
                prefix=prefix,
                sharded_views_original_shape=new_sharded_views_dict
            )
        elif isinstance(state, _FSDPState):
            assert state._use_orig_params, (
                "Setting-up parameter groups requires that all FSDP instances "
                "use use_orig_params=True."
            )
            sharded_views_original_shape_no_prefix = match_and_strip_prefix(
                FSDP_PREFIX, sharded_views_original_shape
            )
            assert (
                len(sharded_views_original_shape_no_prefix)
                == len(sharded_views_original_shape)
            )
            for handle in state._handles:
                for (param_name, _, submodule_name), shape in zip(
                    handle.flat_param._param_infos, handle.flat_param._shapes
                ):
                    fqn = get_fqn(submodule_name, param_name)
                    assert fqn not in sharded_views_original_shape_no_prefix
                    sharded_views_original_shape_no_prefix[fqn] = shape
                for (
                    param_name, _, submodule_name,
                    prim_param_name, _, prim_submodule_name,
                ) in handle.flat_param._shared_param_infos:
                    fqn = get_fqn(submodule_name, param_name)
                    prim_fqn = get_fqn(prim_submodule_name, prim_param_name)
                    assert fqn not in sharded_views_original_shape_no_prefix
                    sharded_views_original_shape_no_prefix[fqn] = (
                        sharded_views_original_shape_no_prefix[prim_fqn]
                    )
            dfs_find_params_and_clean_names(
                state._fsdp_wrapped_module, prefix,
                sharded_views_original_shape_no_prefix,
            )
        else:
            assert state is None, (f"Unknown state type: {type(state)}")
            for name, param in module.named_parameters(recurse=False):
                fqn = get_fqn(prefix, name)
                clean_name_to_real_param_dict[fqn] = param
                if name in sharded_views_original_shape:
                    meta_param = torch.zeros(
                        sharded_views_original_shape[name],
                        dtype=param.dtype,
                        device="meta",
                        requires_grad=param.requires_grad
                    )
                else:
                    meta_param = torch.zeros_like(
                        param, device="meta", requires_grad=param.requires_grad
                    )
                clean_name_to_meta_param_dict[fqn] = meta_param
    
            for name, submodule in module.named_children():
                dfs_find_params_and_clean_names(
                    submodule, prefix=get_fqn(prefix, name),
                    sharded_views_original_shape=match_and_strip_prefix(
                        name + ".", sharded_views_original_shape
                    )
                )

    dfs_find_params_and_clean_names(model, "", {})

    prefix_to_params: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, meta_param in clean_name_to_meta_param_dict.items():
        matched_prefix = ""
        for prefix in _LAYERWISE_PARAM_GROUP_FUNCS:
            if key.startswith(prefix) and len(prefix) > len(matched_prefix):
                matched_prefix = prefix
        if matched_prefix not in prefix_to_params:
            prefix_to_params[matched_prefix] = {}
        key_without_prefix = key[len(matched_prefix):]
        prefix_to_params[matched_prefix][key_without_prefix] = meta_param

    # lr_groups[0] will be the default group (i.e., no lr decay).
    # Put param names in the groups; they will be substituted with the real
    # params later.
    lr_groups: List[Dict[str, Any]] = [{"params": [], "lr": base_lr}]
    for prefix, meta_param_dict in prefix_to_params.items():
        no_layer_wise_lr_decay = (
            layer_wise_lr_decay is None or (
                isinstance(layer_wise_lr_decay, dict)
                and prefix in layer_wise_lr_decay
            )
        )
        if no_layer_wise_lr_decay:
            lr_groups[0]["params"].extend([
                prefix + key for key in meta_param_dict.keys()
            ])
            continue

        lr_decay_factor = (
            layer_wise_lr_decay[prefix]
            if isinstance(layer_wise_lr_decay, dict) else
            layer_wise_lr_decay
        )
        grouping_func = _LAYERWISE_PARAM_GROUP_FUNCS[prefix]
        blockwise_groups = grouping_func(meta_param_dict)
        all_params = set(meta_param_dict.keys())
        for i, group in enumerate(blockwise_groups[::-1]):
            for name in group:
                all_params.remove(name)
            group_with_prefix = [prefix + key for key in group]
            if i == 0:
                lr_groups[0]["params"].extend(group_with_prefix)
            else:
                lr_groups.append({"params": group_with_prefix,
                                  "lr": base_lr * (lr_decay_factor ** i)})
        lr_groups[0]["params"].extend([prefix + key for key in all_params])

    def default_no_wd_criterion(name: str) -> bool:
        return (
            bias_and_1d_params_no_decay and (
                name.endswith(".bias")
                or clean_name_to_meta_param_dict[name].ndim <= 1
            )
        )

    # Split weight decay and no weight decay into 2 groups.
    lr_and_wd_groups: List[Dict[str, Any]] = []
    for group in lr_groups:
        lr = group["lr"]
        wd_group = {"params": [], "lr": lr, "weight_decay": base_weight_decay}
        no_wd_group = {"params": [], "lr": lr, "weight_decay": 0.}
        for param in group["params"]:
            if param in no_weight_decay_list or default_no_wd_criterion(param):
                no_wd_group["params"].append(param)
            else:
                wd_group["params"].append(param)
        lr_and_wd_groups.append(wd_group)
        lr_and_wd_groups.append(no_wd_group)

    # Final cleaning: remove empty groups and not-requiring-grad params.
    final_groups = []
    for group in lr_and_wd_groups:
        group["params"] = [
            param for param in group["params"]
            if clean_name_to_meta_param_dict[param].requires_grad
        ]
        if len(group["params"]) > 0:
            final_groups.append(group)

    # Print grouping information.
    print("Creating the following parameter groups:")
    for i, group in enumerate(final_groups):
        print(f"Parameter Group {i}, learning rate {group['lr']:.7f}, "
              f"weight decay {group['weight_decay']:.7f}, containing:")
        for param in group["params"]:
            print(f"    {param}")
        print()

    # Substitute with real params
    for group in final_groups:
        group["params"] = [
            clean_name_to_real_param_dict[param] for param in group["params"]
        ]

    return final_groups
