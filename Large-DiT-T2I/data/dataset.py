from abc import ABC, abstractmethod
import copy
import json
import logging
import os
from pathlib import Path
import random
from time import sleep
import traceback
import warnings

import h5py
import torch.distributed as dist
from torch.utils.data import Dataset
import yaml


logger = logging.getLogger(__name__)


class DataBriefReportException(Exception):
    def __init__(self, message=None):
        self.message = message

    def __str__(self):
        return f'{self.__class__}: {self.message}'


class ItemProcessor(ABC):
    @abstractmethod
    def process_item(self, data_item, training_mode=False):
        raise NotImplementedError


class MyDataset(Dataset):
    def __init__(self,
                 config_path,
                 item_processor: ItemProcessor,
                 cache_on_disk=False):
        logger.info(f"read dataset config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        logger.info("DATASET CONFIG:")
        logger.info(self.config)

        self.cache_on_disk = cache_on_disk
        if self.cache_on_disk:
            cache_dir = self._get_cache_dir(config_path)
            if dist.get_rank() == 0:
                self._collect_annotations_and_save_to_cache(cache_dir)
            dist.barrier()
            ann, group_indice_range = self._load_annotations_from_cache(cache_dir)
        else:
            cache_dir = None
            ann, group_indice_range = self._collect_annotations()

        self.ann = ann
        self.group_indices = {key: list(range(val[0], val[1])) for key, val in group_indice_range.items()}

        logger.info(f"total length: {len(self)}")

        self.item_processor = item_processor

    def __len__(self):
        return len(self.ann)

    def _collect_annotations(self):
        group_ann = {}
        for meta in self.config['META']:
            meta_path, meta_type = meta['path'], meta.get('type', "default")
            meta_ext = os.path.splitext(meta_path)[-1]
            if meta_ext == ".json":
                with open(meta_path) as f:
                    meta_l = json.load(f)
            elif meta_ext == ".jsonl":
                meta_l = []
                with open(meta_path) as f:
                    for i, line in enumerate(f):
                        try:
                            meta_l.append(json.loads(line))
                        except json.decoder.JSONDecodeError as e:
                            logger.error(f"Error decoding the following jsonl line ({i}):\n{line.rstrip()}")
                            raise e
            else:
                raise NotImplementedError(
                    f"Unknown meta file extension: \"{meta_ext}\". "
                    f"Currently, .json, .jsonl are supported. "
                    "If you are using a supported format, please set the file extension so that the proper parsing "
                    "routine can be called."
                )
            logger.info(f"{meta_path}, type{meta_type}: len {len(meta_l)}")
            if "ratio" in meta:
                random.seed(0)
                meta_l = random.sample(meta_l, int(len(meta_l) * meta['ratio']))
                logger.info(f"sample (ratio = {meta['ratio']}) {len(meta_l)} items")
            if "root" in meta:
                for item in meta_l:
                    if "image" in item:
                        item['image'] = str(Path(meta['root']) / item['image'])
            if meta_type not in group_ann:
                group_ann[meta_type] = []
            group_ann[meta_type] += meta_l

        ann = sum(list(group_ann.values()), start=[])

        group_indice_range = {}
        start_pos = 0
        for meta_type, meta_l in group_ann.items():
            group_indice_range[meta_type] = [start_pos, start_pos + len(meta_l)]
            start_pos = start_pos + len(meta_l)

        return ann, group_indice_range

    def _collect_annotations_and_save_to_cache(self, cache_dir):
        if (Path(cache_dir) / 'data.h5').exists() and (Path(cache_dir) / 'ready').exists():
            # off-the-shelf annotation cache exists
            warnings.warn(
                f"Use existing h5 data cache: {Path(cache_dir)}\n"
                f"Note: if the actual data defined by the data config has changed since your last run, "
                f"please delete the cache manually and re-run this experiment, or the data actually used "
                f"will not be updated"
            )
            return

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        ann, group_indice_range = self._collect_annotations()

        # when cache on disk, rank0 saves items to an h5 file
        serialized_ann = [json.dumps(_) for _ in ann]
        logger.info(f"start to build data cache to: {Path(cache_dir)}")
        with h5py.File(Path(cache_dir) / 'data.h5', 'w') as file:
            dt = h5py.vlen_dtype(str)
            h5_ann = file.create_dataset("ann", (len(serialized_ann),), dtype=dt)
            h5_ann[:] = serialized_ann
            file.create_dataset("group_indice_range", data=json.dumps(group_indice_range))
        with open(Path(cache_dir) / 'ready', 'w') as f:
            f.write("ready")
        logger.info(f"data cache built")

    @ staticmethod
    def _get_cache_dir(config_path):
        config_identifier = config_path
        disallowed_chars = ['/', '\\', '.', '?', '!']
        for _ in disallowed_chars:
            config_identifier = config_identifier.replace(_, '-')
        cache_dir = f"./accessory_data_cache/{config_identifier}"
        return cache_dir

    @ staticmethod
    def _load_annotations_from_cache(cache_dir):
        while not (Path(cache_dir) / 'ready').exists():
            # cache has not yet been completed by rank 0
            assert dist.get_rank() != 0
            sleep(1)
        cache_file = h5py.File(Path(cache_dir) / 'data.h5', 'r')
        annotations = cache_file['ann']
        group_indice_range = json.loads(cache_file['group_indice_range'].asstr()[()])
        return annotations, group_indice_range

    def get_item_func(self, index):
        data_item = self.ann[index]
        if self.cache_on_disk:
            data_item = json.loads(data_item)
        else:
            data_item = copy.deepcopy(data_item)

        return self.item_processor.process_item(data_item, training_mode=True)

    def __getitem__(self, index):
        try:
            return self.get_item_func(index)
        except Exception as e:
            if isinstance(e, DataBriefReportException):
                logger.info(e)
            else:
                logger.info(
                    f"Item {index} errored, annotation:\n"
                    f"{self.ann[index]}\n"
                    f"Error:\n"
                    f"{traceback.format_exc()}"
                )
            for group_name, indices_this_group in self.group_indices.items():
                if indices_this_group[0] <= index <= indices_this_group[-1]:
                    if index == indices_this_group[0]:
                        new_index = indices_this_group[-1]
                    else:
                        new_index = index - 1
                    return self[new_index]
            raise RuntimeError

    def groups(self):
        return list(self.group_indices.values())
