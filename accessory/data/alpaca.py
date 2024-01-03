import warnings
from typing import List, Dict, Optional, Iterator, Tuple
from pathlib import Path
from time import sleep
import h5py
import random
import torch
import yaml
from PIL import Image
import json
import pandas as pd
from accessory.model.tokenizer import Tokenizer
import copy
import numpy as np
import os
from torch.utils.data import Sampler, Dataset
from .system_prompt import format_prompt


class FinetuneDataset(Dataset):
    def __init__(self, config_path, transform, max_words=30, image_words=257, tokenizer=None,
                 cache_on_disk=False, rank=0):

        print(f"read dataset config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)


        self.cache_on_disk = cache_on_disk
        if cache_on_disk:
            # save data items on disk to avoid duplicating annotations in each rank,
            # which could cause a hugh waste of CPU memory
            config_identifier = config_path
            disallowed_chars = ['/', '\\', '.', '?', '!']
            for _ in disallowed_chars:
                config_identifier = config_identifier.replace(_, '-')
            self.cache_dir = f"./accessory_data_cache/{config_identifier}"
            if rank == 0:
                Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None


        # determine if the dataset need to collect annotations from meta files in self.config
        # the collection is needed when:
        # -
        #   cache_on_disk is False, so every rank collects and stores the annotations independently, OR
        # -
        #   cache_on_disk is true & rank == 0 & no off-the-shelf annotation cache, e.g. those created by
        #   prior experiments and runs, exists.
        if not cache_on_disk:
            need_collect_anno = True
        else:
            if rank != 0 :
                need_collect_anno = False
            else:
                if (Path(self.cache_dir)/'data.h5').exists() and (Path(self.cache_dir)/'ready').exists():
                    need_collect_anno = False  # off-the-shelf annotation cache exists
                    print(f"Use existing h5 data cache: {Path(self.cache_dir)}\n"
                          f"Note: if the actual data defined by {config_path} has changed since your last run, "
                          f"please delete the cache manually and re-run this expeirment, or the data actually used "
                          f"will not be updated")
                else:
                    need_collect_anno = True


        if need_collect_anno:
            group_ann = {}
            for meta in self.config['META']:
                meta_path, meta_type = meta['path'], meta['type']
                meta_ext = os.path.splitext(meta_path)[-1]
                #   read data meta file
                #   meta_l should finally be a list of data items, and each data item should be a dict
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
                                print(f"Error decoding the following jsonl line ({i}):\n{line.rstrip()}", force=True)
                                raise e
                elif meta_ext == ".csv":
                    with open(meta_path) as f:
                        chunk = pd.read_csv(meta_path, sep='\t', engine="pyarrow")
                        meta_l = chunk.to_dict(orient="record")
                else:
                    raise NotImplementedError(
                        f"Unknown meta file extension: \"{meta_ext}\". "
                        f"Currently, .json, .jsonl, and .csv files are supported. "
                        "If you are using a supported format, please set the file extension so that the proper parsing "
                        "routine can be called."
                    )

                if meta.get("preprocess", None) is not None:
                    meta_l = MetaPreprocessor().preprocess(meta_l, meta['preprocess'])

                prompt_type = meta.get('prompt_type', 'alpaca')
                print(f"system prompt: {prompt_type}")
                for _ in meta_l:
                    _['sys_prompt'] = prompt_type

                if meta_type not in group_ann:
                    group_ann[meta_type] = []
                print(f"{meta_path}, type{meta_type}: len {len(meta_l)}")
                group_ann[meta_type] += meta_l

            # sort group_ann for higher efficiency (items in one global batch with similar length)
            for meta_type, meta_l in group_ann.items():
                meta_l.sort(
                    key=lambda data_item: len(format_prompt(data_item, data_item["sys_prompt"]) + data_item['output'])
                )

            ann = sum(list(group_ann.values()), start=[])
            group_indice_range = {}
            start_pos = 0
            for meta_type, meta_l in group_ann.items():
                group_indice_range[meta_type] = [start_pos, start_pos + len(meta_l)]
                start_pos = start_pos + len(meta_l)

            if not cache_on_disk:
                self.ann = ann
                self.group_indices = {key: list(range(val[0], val[1])) for key, val in group_indice_range.items()}
            else:
                serialized_ann = [json.dumps(_) for _ in ann]
                print(f"start to build data cache to: {Path(self.cache_dir)}")
                with h5py.File(Path(self.cache_dir)/'data.h5', 'w') as file:
                    dt = h5py.vlen_dtype(str)
                    h5_ann = file.create_dataset("ann", (len(serialized_ann),), dtype=dt)
                    h5_ann[:] = serialized_ann
                    file.create_dataset("group_indice_range", data=json.dumps(group_indice_range))
                with open(Path(self.cache_dir)/'ready', 'w') as f:
                    f.write("ready")
                print(f"data cache built")

        if self.cache_on_disk:
            while not (Path(self.cache_dir)/'ready').exists():
                # cache has not yet been completed by rank 0
                assert rank != 0
                sleep(1)
            cache_file = h5py.File(Path(self.cache_dir) / 'data.h5', 'r')
            self.ann = cache_file['ann']
            group_indice_range = json.loads(cache_file['group_indice_range'].asstr()[()])
            self.group_indices = {key: list(range(val[0], val[1])) for key, val in group_indice_range.items()}


        print(f"total length: {len(self)}")
        self.transform = transform
        self.max_words = max_words
        self.image_words = image_words
        if isinstance(tokenizer, str):
            self.tokenizer = Tokenizer(model_path=tokenizer)
        else:
            self.tokenizer = copy.deepcopy(tokenizer)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        data_item = self.ann[index]
        if self.cache_on_disk:
            data_item = json.loads(data_item)

        image = data_item.get("image", None)
        if image is not None:
            image = Image.open(image).convert('RGB')
            # warnings.warn("image channel format: BGR")
            # image = Image.fromarray(cv2.imread(image))
            image = self.transform(image)
        answer = data_item["output"]

        input1 = format_prompt(data_item, data_item["sys_prompt"])
        input2 = input1 + answer
        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        if image is not None:
            max_words = self.max_words - self.image_words
        else:
            max_words = self.max_words

        padding = max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:max_words]
            warnings.warn(f'Warning for truncation input!\n{data_item}')
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        if image is None:
            return input2, labels, input2_mask
        else:
            return input2, labels, input2_mask, image


    def groups(self):
        return list(self.group_indices.values())


class MetaPreprocessor:
    def __init__(self):
        self.routing = {
            "single_turn_llava": self._preprocess_single_turn_llava,
            "caption": self._preprocess_caption
        }

    def preprocess(self, meta_l:List[Dict], recipe: str):
        return self.routing[recipe](meta_l)

    @ staticmethod
    def _preprocess_single_turn_llava(meta_l: List[Dict]):
        new_meta = []
        for data_item in meta_l:
            new_meta.append({
                "image": data_item['image'],
                "instruction": data_item['conversations'][0]['value'],
                "output": data_item['conversations'][1]['value']
            })
        return new_meta

    @ staticmethod
    def _preprocess_caption(meta_l: List[Dict]):
        new_meta = []
        for data_item in meta_l:
            caption = data_item['caption']
            if isinstance(caption, list):
                caption = random.choice(caption)
            new_meta.append({
                "image": data_item['url'],
                "output": caption
            })

        return new_meta


class FinetuneDistSampler(Sampler):
    #   Distrubuted Sampler ensuring data in a batch are of the same type (e.g. text, image-text)
    def __init__(self, dataset: FinetuneDataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, batch_size = None, acc_grad=1) -> None:
        if num_replicas is None or rank is None or rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid num_replicas ({num_replicas}) or rank ({rank})")
        assert batch_size is not None
        self.batch_size = batch_size

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.acc_grad = acc_grad
        self.epoch = 0
        self.start_iter = 0

        group_indices = dataset.groups()
        global_bsz = batch_size * num_replicas * acc_grad
        len_groups = [len(_) // global_bsz * global_bsz for _ in group_indices]
        group_indices = [indices[:len_indices] for indices, len_indices in zip(group_indices, len_groups)]
        group_n_batch = [len(_)//batch_size for _ in group_indices]
        assert all([_%num_replicas==0 for _ in group_n_batch])
        n_total_batch = sum(group_n_batch)

        assert n_total_batch % self.num_replicas == 0

        self.group_indices = group_indices

        self.total_size = n_total_batch * batch_size
        self.num_samples = self.total_size // num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator:
        global_batch_size = self.batch_size * self.num_replicas * self.acc_grad
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            # self.group_indices should not be changed during shuffle. Only change copy.
            group_indices_shuffle = copy.deepcopy(self.group_indices)
            # for _ in group_indices_shuffle:
            #     rng.shuffle(_)
            global_batched_indices = [
                indices_in_group[i:i+global_batch_size]
                for indices_in_group in group_indices_shuffle
                for i in range(0, len(indices_in_group), global_batch_size)]
            rng.shuffle(global_batched_indices)
            indices = [_ for batch_indices in global_batched_indices for _ in batch_indices]
        else:
            group_indices = copy.deepcopy(self.group_indices)
            indices = [_ for indices_in_group in group_indices for _ in indices_in_group]

        assert len(indices) == self.total_size

        own_indices = []
        for start_pos in range(self.rank * self.batch_size, len(indices), self.num_replicas * self.batch_size):
            own_indices += indices[start_pos: start_pos + self.batch_size]
        # subsample
        assert len(own_indices) == self.num_samples

        if self.start_iter * self.batch_size > len(own_indices):
            own_indices = []
        else:
            own_indices = own_indices[self.start_iter * self.batch_size:]

        return iter(own_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int, start_iter: int = 0) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
            start_iter (int): start iter number.
        """
        self.epoch = epoch
        self.start_iter = start_iter

