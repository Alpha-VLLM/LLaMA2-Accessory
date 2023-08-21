import torch
import yaml
from torch.utils.data import Dataset
from PIL import Image
import json
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


def format_prompt(instruction, input=None):
    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    }
    if input is None or input=='':
        return PROMPT_DICT['prompt_no_input'].format_map({'instruction': instruction})
    else:
        return PROMPT_DICT["prompt_input"].format_map({'instruction': instruction, 'input': input})


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
        for meta_path, meta_type in self.config['META']:
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
                            print(f"Error decoding the following jsonl line ({i}):\n{line.rstrip()}", force=True)
                            raise e
            else:
                raise NotImplementedError(
                    f"Unknown meta file extension: \"{meta_ext}\". Currently, .json and .jsonl files are supported. "
                    "If you are using a supported format, please set the file extension so that the proper parsing "
                    "routine can be called."
                )
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
        data_item = self.ann[index]
        if 'image' in data_item.keys():
            filename = data_item['image']
            question = data_item['conversations'][0]['value']
            answer = data_item['conversations'][1]['value']

            image = Image.open(filename).convert('RGB')
            image = self.transform(image)
            format_instruction = question
            format_input = None
        else:
            image = None
            format_instruction = data_item['instruction'],
            format_input = data_item['input']
            answer = data_item['output']
        input1 = format_prompt(format_instruction, format_input)
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


import math
from typing import TypeVar, Optional, Iterator
from torch.utils.data import Sampler, Dataset

class FinetuneDistSampler(Sampler):
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
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            # self.group_indices should not be changed during shuffle. Only change copy.
            group_indices_shuffle = copy.deepcopy(self.group_indices)
            for _ in group_indices_shuffle:
                rng.shuffle(_)
            global_batched_group_indices = [
                [_[i:i+self.batch_size * self.num_replicas * self.acc_grad]
                 for i in range(0, len(_), self.batch_size * self.num_replicas * self.acc_grad)]
                for _ in group_indices_shuffle]
            global_batched_indices = sum(global_batched_group_indices, start=[])
            rng.shuffle(global_batched_indices)
            indices = sum(global_batched_indices, start=[])
        else:
            group_indices = copy.deepcopy(self.group_indices)
            indices = sum(group_indices, start=[])

        assert len(indices) == self.total_size

        own_indices = []
        for start_pos in range(self.rank * self.batch_size, len(indices), self.num_replicas * self.batch_size):
            own_indices += indices[start_pos: start_pos + self.batch_size]
        # subsample
        assert len(own_indices) == self.num_samples

        return iter(own_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
