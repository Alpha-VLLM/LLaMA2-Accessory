import warnings
from typing import List, Dict, Optional, Iterator, Tuple
import random
import torch
import yaml
from PIL import Image
import json
import pandas as pd
from model.tokenizer import Tokenizer
import copy
import numpy as np
import os
from torch.utils.data import Sampler, Dataset
from .system_prompt import format_prompt

IGNORE_INDEX = -100


class FinetuneDataset(Dataset):
    def __init__(self, config_path, transform, max_words, media_words:Dict, tokenizer_path=None):
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

        # sort group_ann for higher efficiency (items in one global batch with similar length)
        for meta_type, meta_l in group_ann.items():
            meta_l.sort(
                key=lambda data_item: len(format_prompt(data_item, data_item["sys_prompt"]) + data_item['output'])
            )  # todo more accurate sequence length estimation with flexible # of media, or maybe fix it with meta-type

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
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

        self.d_media_symbol_words = copy.deepcopy(media_words)
        print(f"Number of occupied tokens for each media symbol:\n{self.d_media_symbol_words}")
        self.d_media_token_words = {}
        self.d_media_symbol_token = {}
        self.d_media_token_symbol = {}
        self.tokenizer.tokenizer.add_tokens(list(self.d_media_symbol_words.keys()))
        for media_symbol in self.d_media_symbol_words:
            media_token = self.tokenizer.encode(media_symbol, bos=False, eos=False)[0]
            self.d_media_symbol_token[media_symbol] = media_token
            self.d_media_token_words[media_token] = self.d_media_symbol_words[media_symbol]
            self.d_media_token_symbol[media_token] = media_symbol

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        data_item: dict = self.ann[index]

        d_media = {}

        for media_symbol in self.d_media_symbol_words:
            if media_symbol in data_item:
                l_media_path = data_item[media_symbol]  # a list of media paths
            elif media_symbol.lstrip("<").rstrip(">") in data_item:
                l_media_path = data_item[media_symbol.lstrip("<").rstrip(">")]
            else:
                l_media_path = []
            if not isinstance(l_media_path, list):  # data with only one media, in format {"image": image_name, ...}
                l_media_path = [l_media_path]

            d_media[media_symbol] = []
            for media_path in l_media_path:
                image = Image.open(media_path).convert('RGB')
                # warnings.warn("image channel format: BGR")
                # image = Image.fromarray(cv2.imread(image))
                image = self.transform(image)
                d_media[media_symbol].append(image)

        answer = data_item["output"]

        input1 = format_prompt(data_item, data_item["sys_prompt"])
        input2 = input1 + answer
        input1 = self.tokenizer.encode(input1, bos=True, eos=False)
        input2 = self.tokenizer.encode(input2, bos=True, eos=True)
        labels = [IGNORE_INDEX for _ in input2]
        labels[len(input1):] = input2[len(input1):]

        def reserve_space_for_media(tokens, labels, _d_media):
            # check if the number of media symbols is the same as the number of given media resouces.
            for media_symbol in self.d_media_symbol_words:
                l_media = _d_media[media_symbol]
                media_token = self.d_media_symbol_token[media_symbol]
                media_token_count = tokens.count(media_token)
                if media_token_count > 0:
                    # if media symbols already exist in dialog data
                    # the number of symbols should equal to number of media
                    assert media_token_count == len(l_media), \
                        f"{media_token_count} {media_symbol} exists in text, but {len(l_media)} actual media are given"
                else:
                    # add media symbols after BOS token
                    tokens = tokens[:1] + [media_token] * len(l_media) + tokens[1:]  # todo may need special logics to support multiple media types
                    labels = labels[:1] + [IGNORE_INDEX] * len(l_media) + labels[1:]

            # convert media token to reserved placeholders
            new_tokens = []
            new_labels = []
            d_media_spans = {media_symbol:[] for media_symbol in self.d_media_symbol_words}
            assert len(tokens) == len(labels)
            for t, l in zip(tokens, labels):
                if t in self.d_media_token_symbol:
                    media_symbol = self.d_media_token_symbol[t]
                    d_media_spans[media_symbol].append((len(new_tokens), len(new_tokens)+self.d_media_token_words[t]))
                    new_tokens = new_tokens + [-2] * self.d_media_token_words[t]
                    new_labels = new_labels + [l] * self.d_media_token_words[t]
                else:
                    new_tokens.append(t)
                    new_labels.append(l)

            return new_tokens, new_labels, d_media_spans

        input2, labels, d_media_span = reserve_space_for_media(input2, labels, d_media)
        input2 = torch.tensor(input2, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)

        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
            labels = torch.cat((labels, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]  # todo avoid truncation within an image span, especially for supporting image generation
            labels = labels[:self.max_words]
            for symbol, l_span in d_media_span.items():
            # avoid truncation within an image span, especially for supporting image generation
                new_l_span = []
                new_l_media = []
                for span, media in zip(l_span, d_media[symbol]):
                    if span[1] <= self.max_words:
                        new_l_span.append(span)
                        new_l_media.append(media)
                d_media_span[symbol] = new_l_span
                d_media[symbol] = new_l_media
            warnings.warn(f'Warning for truncation input!\n{data_item}')

        input2_mask = input2.ge(0)
        labels_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~labels_mask] = 0

        assert len(input2) == len(labels)

        additional_dict = {key: {"data": d_media[key], "span": d_media_span[key]} for key in d_media}
        return input2, labels, additional_dict

    def groups(self):
        return list(self.group_indices.values())

    @staticmethod
    def collate_func(data):
        input, label, additional_dict = [_[0] for _ in data], [_[1] for _ in data], [_[2] for _ in data]
        input = torch.stack(input, dim=0)
        label = torch.stack(label, dim=0)
        additional_dict = {key: [_[key] for _ in additional_dict] for key in additional_dict[0]}
        return input, label, additional_dict


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

