from typing import List, Dict
import random
import torch
import yaml
from torch.utils.data import Dataset
from PIL import Image
import json
import pandas as pd
from model.tokenizer import Tokenizer
import copy
import torchvision.transforms as transforms
import numpy as np
import os


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from .system_prompt import format_prompt

# create data
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC,
                                 antialias=None),  # 3 is bicubic
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])


transform_val = transforms.Compose([
    transforms.Resize(
        224, interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

class FinetuneDataset(Dataset):
    def __init__(self, config_path, transform=transform_train, max_words=30, image_words=257, tokenizer_path=None):
        print(f"read dataset config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
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
        self.group_ann = group_ann
        self.ann = sum(list(self.group_ann.values()), start=[])

        self.group_indices = {}
        start_pos = 0
        for meta_type, meta_l in self.group_ann.items():
            self.group_indices[meta_type] = list(range(start_pos, start_pos + len(meta_l)))
            start_pos = start_pos + len(meta_l)

        print(f"total length: {len(self)}")
        self.transform = transform
        self.max_words = max_words
        self.image_words = image_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        data_item: dict = self.ann[index]
        image = data_item.pop("image", None)
        if image is not None:
            image = Image.open(image).convert('RGB')
            image = self.transform(image)
        answer = data_item.pop("output")

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


import math
from typing import TypeVar, Optional, Iterator
from torch.utils.data import Sampler, Dataset


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
            for _ in group_indices_shuffle:
                rng.shuffle(_)
            global_batched_indices = [
                indices_in_group[i:i+global_batch_size]
                for indices_in_group in group_indices_shuffle
                for i in range(0, len(indices_in_group), global_batch_size)]
            rng.shuffle(global_batched_indices)
            indices = [_ for batch_indices in global_batched_indices for _ in batch_indices]
        else:
            group_indices = copy.deepcopy(self.group_indices)
            indices = [_ for batch_indices in group_indices for _ in batch_indices]

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

