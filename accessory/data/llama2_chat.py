import copy
import json

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

from model.tokenizer import Tokenizer

from .alpaca import transform_train

IGNORE_INDEX = -100

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


class FinetuneLlam2ChatDataset(Dataset):
    def __init__(self, config_path, transform=transform_train, max_words=512, image_words=257, tokenizer_path=None):
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
            filename = data_item['image']
            image = Image.open(filename).convert('RGB')
            image = self.transform(image)
        else:
            image = None

        dialog = data_item["conversations"]
        # add system message
        dialog = [
            {
                "from": "system",
                "value": DEFAULT_SYSTEM_PROMPT,
            }
        ] + dialog
        dialog = [
            {
                "from": dialog[1]["from"],
                "value": B_SYS
                + dialog[0]["value"]
                + E_SYS
                + dialog[1]["value"],
            }
        ] + dialog[2:]
        # tokenize prompt
        input1 = [
                torch.tensor(self.tokenizer.encode(
                    f"{B_INST} {(prompt['value']).strip()} {E_INST} ",
                    bos=True, eos=False), dtype=torch.int64)
                for prompt, answer in zip(
                    dialog[::2],
                    dialog[1::2])
        ]
        # tokenize prompt & answer
        input2 = [
                torch.tensor(self.tokenizer.encode(
                    f"{B_INST} {(prompt['value']).strip()} {E_INST} {(answer['value']).strip()} ",
                    bos=True, eos=True), dtype=torch.int64)
                for prompt, answer in zip(
                    dialog[::2],
                    dialog[1::2])
        ]

        labels = copy.deepcopy(input2)
        # mask prompt tokens for ignoring losses
        for i, (label, input) in enumerate(zip(labels, input1)):
            label[:len(input)] = -1
            labels[i] = label

        input2 = torch.cat(input2)
        labels = torch.cat(labels)
        
        if image is not None:
            max_words = self.max_words - self.image_words
        else:
            max_words = self.max_words
        padding = max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
            labels = torch.cat((labels, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:max_words]
            labels = labels[:max_words]

        
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