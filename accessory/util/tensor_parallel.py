from collections import OrderedDict
import os
import re
from typing import Dict, List, NamedTuple, Tuple, Type

import torch
import torch.nn as nn

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelEmbedding,
)

r"""_MODEL_PARALLEL_MODULES defines a list of module classes that contains tensor-parallel
parameters which may need special handling.

Each item is a pair whose first item is the module class, and the second item is a dictionary
defining along which dim is each of its weights splitted.

_MODEL_PARALLEL_MODULES is defined as a ``list`` instead of a ``dict`` for well-defined matching
priority: The matching process is expected to be in the order defined in the list and exit on the
first match as returned by ``isinstance``. The design is to handle module sub-classing: Any subclass
of the defined classes can also be matched, and any different handling of the subclass should be
defined BEFORE the item of the parent class.

To correctly save and load the checkpoints we expect each newly involved tensor parallel layer
to be registered in this list.
"""
_MODEL_PARALLEL_MODULES: List[Tuple[Type[nn.Module], Dict[str, int]]] = [
    (ColumnParallelLinear, {"weight": 0, "bias": 0}),
    (RowParallelLinear, {"weight": 1, "bias": -1}),
    (ParallelEmbedding, {"weight": 1}),
]

def _tensor_list_max_diff(tensors: List[torch.Tensor]) -> float:
    for tensor in tensors[1:]:
        assert tensor.dtype is tensors[0].dtype and tensor.size() == tensors[0].size()
    
    if tensors[0].is_complex():
        max_diff = 0.
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                max_diff = max(max_diff, (tensors[i] - tensors[j]).abs().max().item())
        return max_diff
    
    if not tensors[0].is_floating_point():
        tensors = [tensor.float() for tensor in tensors]
    max_tensor, min_tensor = tensors[0].clone(), tensors[0].clone()
    for tensor in tensors[1:]:
        max_tensor = torch.maximum(tensor, max_tensor)
        min_tensor = torch.minimum(tensor, min_tensor)
    return (max_tensor - min_tensor).max().item()


def _load_checkpoint_and_merge_ranks(
    ckpt_files: List[str], weight_parallel_dim: Dict[str, int], verbose: bool = False,
) -> OrderedDict[str, torch.Tensor]:
    mp_rank = fs_init.get_model_parallel_rank()
    mp_world_size = fs_init.get_model_parallel_world_size()
    ckpt_world_size = len(ckpt_files)

    assert ckpt_world_size % mp_world_size == 0
    local_num_shards = ckpt_world_size // mp_world_size
    local_shard_st = local_num_shards * mp_rank
    local_shard_ed = local_num_shards * (mp_rank + 1)
    ckpt_shards = []
    merged_ckpt = OrderedDict()
    for shard_id in range(local_shard_st, local_shard_ed):
        shard = torch.load(ckpt_files[shard_id], map_location="cpu")
        if "model" in shard and isinstance(shard["model"], dict):
            shard = shard["model"]
        ckpt_shards.append(shard)

    for key in list(ckpt_shards[0].keys()):
        param_shards = [shard[key] for shard in ckpt_shards]
        if key not in weight_parallel_dim:  # non tensor parallel parameter
            max_diff = _tensor_list_max_diff(param_shards)
            if max_diff > 0.:
                print(
                    "WARNING! Found unequal replicas of non-tensor-parallel params: "
                    f"name={key}, ranks={','.join(str(x) for x in range(local_shard_st, local_shard_ed))}, "
                    f"max_diff={max_diff}.",
                    force=True,
                )
            merged_ckpt[key] = param_shards[0]
        else:
            merged_ckpt[key] = torch.cat(param_shards, dim=weight_parallel_dim[key])

        # delete the original weights to avoid 2x memory usage.
        for shard in ckpt_shards:
            del shard[key]
    
    return merged_ckpt


def _load_checkpoint_and_split_rank(
    ckpt_files: List[str], weight_parallel_dim: Dict[str, int], verbose: bool = False,
) -> OrderedDict[str, torch.Tensor]:
    raise NotImplementedError()


def _load_checkpoint_and_redistribute_general(
    ckpt_files: List[str], weight_parallel_dim: Dict[str, int], verbose: bool = False,
) -> OrderedDict[str, torch.Tensor]:
    raise NotImplementedError()


def load_tensor_parallel_model(
    model: nn.Module, path: str, format: str, verbose: bool = False
) -> Tuple[List[str], List[str]]:
    r"""This function loads tensor parallel checkpoints to a model. It handles different formats
    (e.g., saved by different training frameworks or released by different organizations) and potentially
    a change of tensor parallel size (e.g., reducing tensor parallel size when running on fewer GPUs
    each with larger memory).

    Args:
        model (nn.Module): The model to load the checkpoint into.
        path (str): A path containing checkpoint files.
        format (str): Format of the checkpoing files. Supported formats: ``consolidated`` (saved by our
            framework) and ``meta_ori`` (original checkpoints released in Meta's LLaMA repo).
        verbose (bool): Print verbose information about the loading process for debug purposes.
            Default=``False``.
    """

    def print_if_verbose(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    weight_parallel_dim = {}
    for name, module in model.named_modules():
        for class_, dict_ in _MODEL_PARALLEL_MODULES:
            if isinstance(module, class_):
                for leaf_name, dim in dict_.items():
                    full_name = name + "." + leaf_name if name else leaf_name
                    if dim >= 0:
                        weight_parallel_dim[full_name] = dim
                break

    mp_world_size = fs_init.get_model_parallel_world_size()

    if format in ["meta_ori", "consolidated"]:
        # meta_ori and consolidated are essentially the same format: Both store weights
        # of each model parallel rank in a separate file. The minor differences are:
        # 1. In "meta_ori" format, filenames contain only model_parallel_rank but in 
        #    "consolidated" format, filenames also contain model_parallel_world_size to
        #    make a missing part of the checkpoints instantly noticeable.
        # 2. In "consolidated" format, state keys additionally contain the "llma." prefix.

        # Integrity check and checkpoint mp_world_size calculation if needed.
        if format == "meta_ori":
            pattern = re.compile("^consolidated.(\d{2}).pth$")
        else:
            pattern = re.compile("^consolidated.(\d{2})-of-(\d{2}).model.pth$")
        ckpt_fns = [fn for fn in os.listdir(path) if pattern.match(fn)]
        ckpt_mp_world_size = len(ckpt_fns)
        assert ckpt_mp_world_size > 0, (
            f"\"{path}\" is not a valid {format} format checkpoint path: "
            "No file with valid name is found in the path."
        )
        ckpt_files = []
        for i in range(ckpt_mp_world_size):
            if format == "meta_ori":
                fn = f"consolidated.{i:02d}.pth"
            else:
                fn = f"consolidated.{i:02d}-of-{ckpt_mp_world_size:02d}.model.pth"
            full_path = os.path.join(path, fn)
            assert os.path.isfile(full_path), f"\"{full_path}\" is not a file."
            ckpt_files.append(full_path)
        
        # Dispatch to different implementations for better performance: Shorten the start-up
        # time as much as possible because we strive for better user experience!
        if ckpt_mp_world_size % mp_world_size == 0:
            local_state_dict = _load_checkpoint_and_merge_ranks(
                ckpt_files, weight_parallel_dim, verbose
            )
        elif mp_world_size % ckpt_mp_world_size == 0:
            local_state_dict = _load_checkpoint_and_split_rank(
                ckpt_files, weight_parallel_dim, verbose
            )
        else:
            local_state_dict = _load_checkpoint_and_redistribute_general(
                ckpt_files, weight_parallel_dim, verbose
            )
        
        if format == "meta_ori":
            local_state_dict = OrderedDict(
                ("llma." + key, value) for key, value in local_state_dict.items()
            )
        
        return model.load_state_dict(local_state_dict, strict=False)

    else:
        raise NotImplementedError(f"Checkpoint format {format} is unknown.")
