import torch
import yaml
from torch.utils.data import Dataset
from PIL import Image
import json
from model.tokenizer import Tokenizer
import copy
from ..alpaca import transform_train
import os
import numpy as np

from . import lib as conversation_lib

IGNORE_INDEX = -100

DEFAULT_IMAGE_TOKEN = "<image>"
try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _tokenize_fn(strings,
                 tokenizer):
    """Tokenize a list of strings."""
    tokenized_list = [
        torch.tensor(tokenizer.encode(text, bos=True, eos=False), dtype=torch.int64) for text in strings
    ]

    input_ids = labels = [
        tokenized for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.ne(-1).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers, s_ids=None):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    s_ids = s_ids[1:]
    target[:cur_idx] = IGNORE_INDEX

    for tokenized_len, speaker, s_id in zip(tokenized_lens, speakers, s_ids):
        if cur_idx >= target.shape[0]:
            break
        tmp = target[cur_idx + 2:cur_idx + tokenized_len]
        if not torch.equal(tmp,
                           s_id[2:2 + len(tmp)]):
            print("a sentence mismatches the corresponding piece "
                  "in the conversation")

        if speaker == "human":
            target[cur_idx + 2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'

        if DEFAULT_IMAGE_TOKEN in sentence["value"]:
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, '').strip()

        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


class FinetuneDialogDataset(Dataset):
    def __init__(self, config_path, transform=transform_train, max_words=30, image_words=257, tokenizer_path=None):
        print(f"read dataset config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        group_ann = {}
        for meta_path, meta_type in self.config['META']:
            meta_l = json.load(open(meta_path))
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
            filename = os.path.join('/data0/data/coco/train2017', data_item['image'])
            image = Image.open(filename).convert('RGB')
            image = self.transform(image)
        else:
            image = None

        source = data_item["conversations"]
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)


        input2 = torch.tensor(self.tokenizer.encode(conversation, bos=True, eos=True), dtype=torch.int64)
        concat_sentence = [header] + [s["value"] for s in source]

        tokenized_sentence = _tokenize_fn(concat_sentence,
                                          self.tokenizer)
        tokenized_lens = tokenized_sentence["input_ids_lens"]
        tokenized_ids = tokenized_sentence["input_ids"]

        if image is not None:
            max_words = self.max_words - self.image_words
        else:
            max_words = self.max_words
        padding = max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:max_words]
        speakers = [sentence["from"] for sentence in source]
        labels = copy.deepcopy(input2)
        _mask_targets(labels, tokenized_lens, speakers, tokenized_ids)


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