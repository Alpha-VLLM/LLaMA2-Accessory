from collections import OrderedDict
import os
import re
from typing import Dict, List, Set, Tuple, Type

import torch
import torch.nn as nn

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelEmbedding,
)

import regex as re

# _MODEL_PARALLEL_MODULES defines a list of module classes that contains
# tensor-parallel parameters which may need special handling.
#
# Each item is a pair whose first item is the module class, and the second
# item is a dictionary defining along which dim is each of its weights
# splitted.
#
# _MODEL_PARALLEL_MODULES is defined as a ``list`` instead of a ``dict`` for
# well-defined matching priority: The matching process is expected to be in
# the order defined in the list and exit on the first match as returned by
# ``isinstance``. The design is to handle module sub-classing: Any subclass
# of the defined classes can also be matched, and any different handling of
# the subclass should be defined BEFORE the item of the parent class.
#
# To correctly save and load the checkpoints we expect each newly involved
# tensor parallel layer to be registered in this list.
_MODEL_PARALLEL_MODULES: List[Tuple[Type[nn.Module], Dict[str, int]]] = [
    (ColumnParallelLinear, {"weight": 0, "bias": 0}),
    (RowParallelLinear, {"weight": 1, "bias": -1}),
    (ParallelEmbedding, {"weight": 1}),
]

FORMAT_FILENAME_PATTERNS: Dict[str, re.Pattern] = {
    "meta_ori": re.compile(r"^consolidated.(\d{2}).pth$"),
    "consolidated": re.compile(r"^consolidated.(\d{2})-of-(\d{2}).model.pth$"),
    "consolidated_diff": re.compile(r"^consolidated.(\d{2})-of-(\d{2}).model-"
                                    r"diff.pth$"),
}


def get_weight_parallel_dim(model: nn.Module):
    weight_parallel_dim = {}
    for name, module in model.named_modules():
        for class_, dict_ in _MODEL_PARALLEL_MODULES:
            if isinstance(module, class_):
                for leaf_name, dim in dict_.items():
                    full_name = name + "." + leaf_name if name else leaf_name
                    if dim >= 0:
                        weight_parallel_dim[full_name] = dim
                break
    return weight_parallel_dim


def _tensor_list_max_diff(tensors: List[torch.Tensor]) -> float:
    for tensor in tensors[1:]:
        assert (tensor.dtype is tensors[0].dtype
                and tensor.size() == tensors[0].size())

    if tensors[0].is_complex():
        max_diff = 0.
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                max_diff = max(max_diff,
                               (tensors[i] - tensors[j]).abs().max().item())
        return max_diff

    if not tensors[0].is_floating_point():
        tensors = [tensor.float() for tensor in tensors]
    max_tensor, min_tensor = tensors[0].clone(), tensors[0].clone()
    for tensor in tensors[1:]:
        max_tensor = torch.maximum(tensor, max_tensor)
        min_tensor = torch.minimum(tensor, min_tensor)
    return (max_tensor - min_tensor).max().item()


def _load_checkpoint_and_merge_ranks(
    model: nn.Module, ckpt_path: str, ckpt_mp_world_size: int,
    verbose: bool, format: str,
) -> OrderedDict[str, torch.Tensor]:
    weight_parallel_dim = get_weight_parallel_dim(model)
    named_parameters = dict(model.named_parameters())

    mp_rank = fs_init.get_model_parallel_rank()
    mp_world_size = fs_init.get_model_parallel_world_size()

    assert ckpt_mp_world_size % mp_world_size == 0
    local_num_shards = ckpt_mp_world_size // mp_world_size
    local_shard_st = local_num_shards * mp_rank
    local_shard_ed = local_num_shards * (mp_rank + 1)
    ckpt_shards = []
    merged_ckpt = OrderedDict()
    for shard_id in range(local_shard_st, local_shard_ed):
        ckpt_shards.append(
            load_tensor_parallel_shard_state_dict(
                ckpt_path, format, shard_id, ckpt_mp_world_size
            )
        )

    for key in list(set([_ for shard in ckpt_shards for _ in shard.keys()])):
        if key not in named_parameters:
            print(f"discard unexpected parameter: {key}")
            continue
        param_shards = [shard[key] for shard in ckpt_shards if key in shard]
        if hasattr(named_parameters[key], "model_parallel_merge"):
            merged_ckpt[key] = named_parameters[key].model_parallel_merge(param_shards)
        elif key in weight_parallel_dim:
            merged_ckpt[key] = torch.cat(param_shards, dim=weight_parallel_dim[key])
        else:  # non tensor parallel parameter
            max_diff = _tensor_list_max_diff(param_shards)
            if max_diff > 0.:
                print("WARNING! Found unequal replicas of non-tensor-parallel "
                      f"params: name={key}, "
                      f"ranks={list(range(local_shard_st, local_shard_ed))}, "
                      f"max_diff={max_diff}.",
                      force=True)
            merged_ckpt[key] = param_shards[0]

        # delete the original weights to avoid 2x memory usage.
        for shard in ckpt_shards:
            if key in shard:
                del shard[key]

    return merged_ckpt


def _load_checkpoint_and_split_rank(
    model: nn.Module, ckpt_path: str, ckpt_mp_world_size: int,
    verbose: bool, format: str,
) -> OrderedDict[str, torch.Tensor]:
    weight_parallel_dim = get_weight_parallel_dim(model)
    named_parameters = dict(model.named_parameters())

    mp_rank = fs_init.get_model_parallel_rank()
    mp_world_size = fs_init.get_model_parallel_world_size()

    assert mp_world_size % ckpt_mp_world_size  == 0
    shard_split_to = mp_world_size // ckpt_mp_world_size
    shard_id = mp_rank // shard_split_to
    split_id = mp_rank % shard_split_to
    ori_ckpt = load_tensor_parallel_shard_state_dict(ckpt_path, format, shard_id, ckpt_mp_world_size)
    split_ckpt = OrderedDict()

    for key, ori_param in ori_ckpt.items():
        if key in named_parameters:
            if hasattr(named_parameters[key], "model_parallel_split"):
                split_ckpt[key] = named_parameters[key].model_parallel_split(ori_param, shard_split_to)[split_id]
            elif key in weight_parallel_dim:
                split_ckpt[key] = torch.chunk(ori_param, shard_split_to, weight_parallel_dim[key])[split_id]
            else:  # non tensor parallel parameter
                split_ckpt[key] = ori_param
        else:
            print(f"discard unexpected parameter: {key}")

    return split_ckpt


def _load_checkpoint_and_redistribute_general(
    model: nn.Module, ckpt_path: str, ckpt_mp_world_size: int,
    verbose: bool, format: str,
) -> OrderedDict[str, torch.Tensor]:
    raise NotImplementedError()


def get_tensor_parallel_shards_file_name(
    format: str, mp_size: int
) -> List[str]:
    r"""A helper function that returns a list of tensor-parallel shard file
    names by format and tensor parallel size.

    Args:
        format (str): Name of the checkpoint format.
        mp_size (int): Tensor parallel size of the checkpoint.

    Returns:
        List[str]: A list of file names with the i-th element being the file
            name of the i-th tensor parallel shard.
    """
    return {
        "meta_ori": [
            f"consolidated.{i:02d}.pth" for i in range(mp_size)
        ],
        "consolidated": [
            f"consolidated.{i:02d}-of-{mp_size:02d}.model.pth"
            for i in range(mp_size)
        ],
        "consolidated_diff": [
            f"consolidated.{i:02d}-of-{mp_size:02d}.model-diff.pth"
            for i in range(mp_size)
        ],
    }[format]


def load_tensor_parallel_shard_state_dict(
    path: str, format: str, shard_id: int, num_shards: int,
) -> Dict[str, torch.Tensor]:
    r"""Load one tensor parallel state dict shard from the disk and post
    process according to format.

    Args:
        path (str): Path to the folder containing the checkpoint shards.
        format (str): Name of the format of the checkpoint.
        shard_id (int): The tensor parallel rank to load.
        num_shards (int): The tensor parallel world size of the checkpoint.

    Returns:
        Dict[str, torch.Tensor]: The loaded and processed state dict.
    """
    shard_path = os.path.join(
        path,
        get_tensor_parallel_shards_file_name(format, num_shards)[shard_id],
    )
    shard = torch.load(shard_path, map_location="cpu")
    if format.startswith("consolidated"):
        if "model" in shard and isinstance(shard["model"], dict):
            shard = shard["model"]
    elif format == "meta_ori":
        shard = dict(("llma." + key, value)
                     for key, value in shard.items())
    return shard


def load_tensor_parallel_model_state_dict(
    model: nn.Module, path: str, format: str, verbose: bool = False
) -> OrderedDict[str, torch.Tensor]:
    r"""This function loads tensor parallel checkpoints and handles
    different formats (e.g., saved by different training frameworks or
    released by different organizations) and potentially a change of tensor
    parallel size (e.g., reducing tensor parallel size when running on fewer
    GPUs each with larger memory).

    Args:
        model (nn.Module): The model to load the checkpoint into.
        path (str): A path containing checkpoint files.
        format (str): Format of the checkpoing files. Supported formats:
            ``consolidated`` (saved by our framework) and ``meta_ori``
            (original checkpoints released in Meta's LLaMA repo).
        verbose (bool): Print verbose information about the loading process
            for debug purposes. Default=``False``.

    Returns:
        OrderedDict[str, torch.Tensor]: The model state_dict local to the
            model parallel rank of the current process.
    """
    def print_if_verbose(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    mp_world_size = fs_init.get_model_parallel_world_size()

    if format in ["meta_ori", "consolidated", "consolidated_diff"]:
        # meta_ori and consolidated are essentially the same format: Both
        # store weights of each model parallel rank in a separate file. The
        # minor differences are:
        # 1. In "meta_ori" format, filenames contain only model_parallel_rank
        #    but in "consolidated" format, filenames also contain
        #    model_parallel_world_size to make a missing part of the
        #    checkpoints instantly noticeable.
        # 2. In "consolidated" format, state keys additionally contain the
        #    "llma." prefix.

        # Integrity check and checkpoint mp_world_size calculation if needed.
        pattern = FORMAT_FILENAME_PATTERNS[format]
        ckpt_fns = [fn for fn in os.listdir(path) if pattern.match(fn)]
        ckpt_mp_world_size = len(ckpt_fns)
        assert ckpt_mp_world_size > 0, (
            f"\"{path}\" is not a valid {format} format checkpoint path: "
            "No file with valid name is found in the path."
        )

        # Dispatch to different implementations for better performance:
        # Shorten the start-up time as much as possible because we strive for
        # better user experience!
        if ckpt_mp_world_size % mp_world_size == 0:
            local_state_dict = _load_checkpoint_and_merge_ranks(
                model, path, ckpt_mp_world_size, verbose, format
            )
        elif mp_world_size % ckpt_mp_world_size == 0:
            local_state_dict = _load_checkpoint_and_split_rank(
                model, path, ckpt_mp_world_size, verbose, format
            )
        else:
            local_state_dict = _load_checkpoint_and_redistribute_general(
                model, path, ckpt_mp_world_size, verbose, format
            )

        return local_state_dict

    else:
        raise NotImplementedError(f"Checkpoint format {format} is unknown.")


def load_tensor_parallel_model(
    model: nn.Module, path: str, format: str, verbose: bool = False
) -> Tuple[List[str], List[str]]:
    r""""This method calls ``load_tensor_parallel_model_state_dict`` (which
    handles multiple formats / unmatched tensor parallel size) and load the
    converted checkpoint into a model.

    Args:
        model (nn.Module): The model to load the checkpoint into.
        path (str): A path containing checkpoint files.
        format (str): Format of the checkpoing files. Supported formats:
            ``consolidated`` (saved by our framework) and ``meta_ori``
            (original checkpoints released in Meta's LLaMA repo).
        verbose (bool): Print verbose information about the loading process
            for debug purposes. Default=``False``.

    Returns:
        Tuple[List[str], List[str]]: Returns two lists of strings, the first
            being the missing keys and the second being the unexpected keys,
            following the same convention as
            ``torch.nn.Module.load_state_dict``.
    """
    assert not format.endswith("_diff"), (
        "A *_diff checkpoint must be used together with the corresponding "
        "base checkpoint to obtain the full model weights."
    )
    local_state_dict = load_tensor_parallel_model_state_dict(
        model, path, format, verbose
    )
    load_result = model.load_state_dict(local_state_dict, strict=False)
    print('load tensor parallel model result:\n', load_result)
    return load_result


def infer_checkpoint_format_and_mp_size(path: str) -> str:
    r"""This method infers the checkpoint format and model parallel size
    according to the files in the given folder.

    Args:
        path (str): The path to be inspected.

    Raises:
        NotImplementedError: If the supplied path is not a folder, or no file
            in the folder belong to any recognized format, or files belong to
            multiple formats are found, or the file names do not match the
            expected list of a given tensor parallel size.

    Returns:
        Tuple[str, int]: A tuple with the first element being the name of the
            format, and the second being the inferred model parallel size.
    """
    if not os.path.isdir(path):
        raise NotImplementedError("The given path does not point to a valid "
                                  "folder.")
    files_in_folder = os.listdir(path)
    files_in_folder = [fn for fn in files_in_folder
                       if os.path.isfile(os.path.join(path, fn))]
    inferred_format, inferred_mp_size = None, None
    for format, pattern in FORMAT_FILENAME_PATTERNS.items():
        matched_fns = [fn for fn in files_in_folder if pattern.match(fn)]
        if matched_fns:
            if inferred_format is None:
                inferred_format = format
                inferred_mp_size = len(matched_fns)
            else:
                raise NotImplementedError(f"Multiple matched format detected: "
                                          f"{inferred_format} and {format}.")
    if inferred_format is None:
        folder_contents = ", ".join(
            [x if os.path.isfile(os.path.join(path, x)) else x + " (not a file)"
             for x in os.listdir(path)]
        )
        raise NotImplementedError(
            f"Files in the given folder do not match  any format. "
            f"Contents in the folder: [{folder_contents}]."
        )

    expected_files_list = get_tensor_parallel_shards_file_name(
        inferred_format, inferred_mp_size
    )
    for fn in expected_files_list:
        if fn not in files_in_folder:
            raise NotImplementedError("An expected file is not found in the "
                                      "target folder: " + fn)

    return inferred_format, inferred_mp_size


def load_diff_checkpoint(
    model: nn.Module, state_dict: Dict[str, torch.Tensor],
    existing_keys: Set[str],
) -> Tuple[List[str], List[str]]:
    r"""This method loads model from a *_diff format checkpoint. The behavior
    of loading a diff checkpoint is different from loading a regular
    checkpoint: In case a key is in the given ``existing_keys``, the new value
    of the tensor is the value in the state_dict plus the old value.

    Note:
        The input ``state_dict`` will be changed in-place to save memory.

    Args:
        model (nn.Module): The model to load the state dict into.
        state_dict (Dict[str, torch.Tensor]): The state dict to be loaded into
            the model.
        existing_keys (Set[str]): A set of keys that have appeared in the
            previous checkpoints. If a key is in this set, the corresponding
            value from the state dict will be added to the value in the model;
            otherwise the value in the model is considered uninitialized and
            is directly set to the value in the state dict.

    Returns:
        Tuple[List[str], List[str]]: A pair of lists including missing keys and
            unexpected keys, following the regular
            ``torch.nn.Module.load_stat_dict``.
    """
    model_state_dict = model.state_dict()
    for key in list(state_dict.keys()):
        if key in existing_keys and key in model_state_dict:
            orig_value = model_state_dict[key]
            diff_value = state_dict[key]
            orig_value = orig_value.to(diff_value.device)
            diff_value = diff_value.to(orig_value.dtype)
            state_dict[key] = orig_value + diff_value
    return model.load_state_dict(state_dict, strict=False)


def load_tensor_parallel_model_list(
    model: nn.Module, path_list: str|List[str], verbose: bool = False
) -> Dict:
    r"""This method accepts a list of checkpoint paths, and load each
    checkpoint to the model in the order as given in the list. The behaviors
    of a base checkpoint format (currently supported: meta_ori, consolidated)
    and a diff checkpoint format (currently supported: consolidated_diff) is
    different: Values in a base checkpoint will override previous values with
    the same key, but values in a diff checkpoint will be added to the previous
    values with the same key. The format of each checkpoint path is inferred
    automatically so no ``format`` argument is needed as in
    ``load_tensor_parallel_model``. The method internally calls
    ``load_tensor_parallel_model_state_dict`` so loading from checkpoints of
    unmatched tensor parallel size is also supported.

    Args:
        model (nn.Module): A PyTorch model to load the checkpoints into.
        path_list (List[str]): A list of checkpoint paths. Each checkpoint is
            loaded in the order as supplied in the list.
        verbose (bool): Whether verbose information should be printed (e.g.,
            for debugging purposes). The default is ``False``.

    Returns:
        Tuple[List[str], List[str]]: Returns two lists of strings, the first
            being the missing keys and the second being the unexpected keys,
            following the same convention as
            ``torch.nn.Module.load_state_dict``. A key is deemed missing if it
            does not occur in any of the checkpoints in the list, and is deemed
            unexpected if it is unexpected to the model and has appeared in any
            one of the checkpoints in the list.
    """
    if isinstance(path_list, str):
        path_list = [path_list]
    existing_keys, missing_keys, unexpected_keys = set(), set(model.state_dict().keys()), set()
    for i, path in enumerate(path_list):
        inferred_format, _ = infer_checkpoint_format_and_mp_size(path)
        print(f"Loading from checkpoint at: {path} ({i + 1} of "
              f"{len(path_list)}, format is \"{inferred_format})\"")
        assert i != 0 or not inferred_format.endswith("_diff"), (
            "The first checkpoint in the list cannot be a *_diff checkpoint."
        )
        state_dict = load_tensor_parallel_model_state_dict(
            model, path, inferred_format, verbose
        )
        if inferred_format.endswith("_diff"):
            step_missing_keys, step_unexpected_keys = load_diff_checkpoint(
                model, state_dict
            )
        else:
            for key in state_dict:
                if key in existing_keys:
                    print(f"A key ({key}) is overrided by a full checkpoint "
                          f"(at {path}).")
            step_missing_keys, step_unexpected_keys = model.load_state_dict(
                state_dict, strict=False
            )
        existing_keys.update(state_dict.keys())
        missing_keys.intersection_update(step_missing_keys)
        unexpected_keys.update(step_unexpected_keys)

    return {"missing_keys": list(missing_keys), "unexpected_keys": list(unexpected_keys)}


def tensor_load_shard(
    target: torch.Tensor, parallel_dim: int, num_shards: int, shard_id: int,
    value: torch.Tensor, mode: str = "set",
) -> None:
    r"""A helper function to partially load a tensor. This can save memory
    sometimes as this allows tensor parallel shards to stream into memory (
    without being concatenated into a full model first).

    Args:
        target (``torch.Tensor``): The target tensor to load the values into.
        parallel_dim (int): Tensor parallel dimension of the tensor.
        num_shards (int): Number of tensor parallel shards of the value.
        shard_id (int): The shard id of the current value.
        value (``torch.Tensor``): The value to be loaded into the target
            tensor.
        mode (str): The supported values are ``set`` and ``add``. If ``set``,
            the old value in the target tensor is overrided with the new value.
            If ``add``, the new value is added to the old value.
    """
    assert parallel_dim < target.ndim or parallel_dim == -1
    target_slices = []
    for i in range(target.ndim):
        if i == parallel_dim:
            dim_st = target.size(i) // num_shards * shard_id
            dim_ed = target.size(i) // num_shards * (shard_id + 1)
            target_slices.append(slice(dim_st, dim_ed))
        else:
            target_slices.append(slice(None))
    if parallel_dim == -1 and shard_id != 0 and mode in ["set", "add"]:
        return
    if mode == "set":
        target[target_slices] = value
    elif mode == "add":
        target[target_slices] += value
    else:
        raise NotImplementedError(f"Unknown mode: {mode}.")


class ShardedTensorLoader:
    r"""A helper class to load a tensor parallel sharded tensor and track the
    loading status (i.e., check if a tensor is loaded with consistent tensor
    parallel size, check that each shard is loaded only once, and check that
    all shards are loaded).
    """

    def __init__(
        self,
        target: torch.Tensor,
        num_shards: int,
        shard_dim: int,
        mode: str = "set",
    ) -> None:
        r"""Initialize a ShardedTensorLoader.

        Args:
            target (``torch.Tensor``): The target tensor where value shards are
                loaded into.
            num_shards (int): Number of expected shards.
            shard_dim (int): The dimension along which the tensor is sharded.
            mode (str): Supported options are ``set`` and ``add``. If ``set``,
                the old value in the target tensor is overrided with the new
                value. If ``add``, the new value is added to the old value.
        """
        self._target = target
        self._num_shards = num_shards
        self._shard_dim = shard_dim
        self._mode = mode

        self._loaded_shards = set()

    def load_shard(self, shard_id: int, value: torch.Tensor) -> None:
        r"""Load a shard into the target tensor.

        Args:
            shard_id (int): The shard id of the current value.
            value (``torch.Tensor``): The value to be loaded.
        """
        assert shard_id not in self._loaded_shards
        assert shard_id >= 0 and shard_id < self._num_shards
        self._loaded_shards.add(shard_id)
        tensor_load_shard(self._target, self._shard_dim, self._num_shards,
                          shard_id, value, self._mode)

    def is_complete(self) -> bool:
        r"""Check if all the shards are loaded to the target tensor."""
        assert all(x >= 0 and x < self._num_shards
                   for x in self._loaded_shards)
        return len(self._loaded_shards) == self._num_shards
