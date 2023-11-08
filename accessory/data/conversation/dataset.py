import copy
import warnings

from typing import Dict
import torch
import yaml
from torch.utils.data import Dataset
from PIL import Image
from ..data_reader import read_img_general
import json
from model.tokenizer import Tokenizer
import os

from . import lib as conversation_lib

import traceback

IGNORE_INDEX = -100


class LabelAllZeroError(Exception):
    def __init__(self, message=None):
        self.message = message

    def __str__(self):
        return f'LabelAllZeroError: {self.message}'


class ConversationGenerator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.header = f"{conversation_lib.default_conversation.system}\n\n"
        self._probe_tokenizer_style()

    def _probe_tokenizer_style(self):
        """
        Given a sentence, e.g. "My darling", some tokenizers will make the space a seperate token,
        while some others will merge the space into the next word, forming a token representing " darling".
        Knowing which style the tokenizer takes is necessary for correct ground-truth label masking.

        """
        probe = "Probe am I"
        sentence1 = self.tokenizer.encode(conversation_lib.default_conversation.roles[1] + ": " + probe,
                                          bos=False, eos=False)
        sentence2 = self.tokenizer.encode(probe,
                                          bos=False, eos=False)
        if sentence1[-len(sentence2):] == sentence2:
            self.space_before_to_predict = False
        else:
            sentence3 = self.tokenizer.encode(" " + probe,
                                              bos=False, eos=False)
            assert sentence1[-len(sentence3):] == sentence3
            self.space_before_to_predict = True

    def add_speaker_and_signal(self, source, get_conversation=True):
        """Add speaker and start/end signal on each round."""
        BEGIN_SIGNAL = "### "
        END_SIGNAL = "\n"
        conversation = self.header

        to_predict_list = []

        for sentence in source:
            from_str = sentence["from"]
            if from_str.lower() in ["human"]:
                from_str = conversation_lib.default_conversation.roles[0]
            elif from_str.lower() in ["gpt", "assistant"]:
                from_str = conversation_lib.default_conversation.roles[1]
            else:
                raise ValueError(f"unknown dialog role: {from_str.lower()}")

            value = sentence["value"]

            sentence_value = BEGIN_SIGNAL + from_str + ": " + value + END_SIGNAL

            if from_str == conversation_lib.default_conversation.roles[1]:
                to_predict_value = value + END_SIGNAL + "###"
                if self.space_before_to_predict:
                    to_predict_value = " " + to_predict_value
                to_predict_list.append(to_predict_value)

            if get_conversation:
                conversation = conversation + sentence_value

        conversation = conversation + BEGIN_SIGNAL
        return conversation, to_predict_list


class FinetuneDialogDataset(Dataset):
    def __init__(self, config_path, transform, max_words, media_words: Dict, tokenizer_path=None):
        print(f"read dataset config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        group_ann = {}
        for meta in self.config['META']:
            meta_path, meta_type = meta['path'], meta['type']
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
                    f"Unknown meta file extension: \"{meta_ext}\". "
                    f"Currently, .json, .jsonl are supported. "
                    "If you are using a supported format, please set the file extension so that the proper parsing "
                    "routine can be called."
                )
            if meta_type not in group_ann:
                group_ann[meta_type] = []
            print(f"{meta_path}, type{meta_type}: len {len(meta_l)}")
            group_ann[meta_type] += meta_l

        # sort group_ann for higher efficiency (items in one global batch with similar length)
        for meta_type, meta_l in group_ann.items():
            meta_l.sort(key=lambda data_item: sum([len(_['value']) for _ in data_item['conversations']]))

        self.group_ann = group_ann
        self.ann = sum(list(self.group_ann.values()), start=[])

        self.group_indices = {}
        start_pos = 0
        for meta_type, meta_l in self.group_ann.items():
            self.group_indices[meta_type] = list(range(start_pos, start_pos + len(meta_l)))
            start_pos = start_pos + len(meta_l)

        print(f"total length: {len(self)}")
        self.transform = transform
        print(f"transform:\n{self.transform}")
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        self.conversation_generator = ConversationGenerator(self.tokenizer)

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

    def get_item_func(self, index):
        data_item = self.ann[index]

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
                image = read_img_general(media_path)
                # warnings.warn("image channel format: BGR")
                # image = Image.fromarray(cv2.imread(image))
                image = self.transform(image)
                d_media[media_symbol].append(image)

        source = data_item["conversations"]
        conversation, to_predict_values = self.conversation_generator.add_speaker_and_signal(source)
        if len(to_predict_values) == 0:
            raise ValueError(f"see dialog data with nothing to predict, data: {data_item}")

        tokenzed_conversation = self.tokenizer.encode(conversation, bos=True, eos=True)
        labels = [IGNORE_INDEX for _ in tokenzed_conversation]

        check_pos = 0
        for value in to_predict_values:
            tokenized_value = self.tokenizer.encode(value, bos=False, eos=False)
            value_pos = find_sublist(tokenzed_conversation[check_pos:], tokenized_value) + check_pos
            if value_pos == -1:
                raise ValueError("a sentence mismatches the corresponding piece in the conversation, data: {data_item}")
            labels[value_pos:value_pos+len(tokenized_value)] = tokenized_value
            assert labels[value_pos:value_pos+len(tokenized_value)] == tokenzed_conversation[value_pos:value_pos+len(tokenized_value)]
            check_pos = value_pos+len(tokenized_value)


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


        tokenzed_conversation, labels, d_media_span = reserve_space_for_media(tokenzed_conversation, labels, d_media)
        tokenzed_conversation = torch.tensor(tokenzed_conversation, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)

        padding = self.max_words - tokenzed_conversation.shape[0]
        if padding > 0:
            tokenzed_conversation = torch.cat((tokenzed_conversation, torch.zeros(padding, dtype=torch.int64) - 1))
            labels = torch.cat((labels, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokenzed_conversation = tokenzed_conversation[:self.max_words]
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

        tokenzed_conversation_mask = tokenzed_conversation.ge(0)
        labels_mask = labels.ge(0)
        tokenzed_conversation[~tokenzed_conversation_mask] = 0
        labels[~labels_mask] = 0

        assert len(tokenzed_conversation) == len(labels)

        if torch.count_nonzero(labels) == 0:
            raise LabelAllZeroError()

        additional_dict = {key: {"data": d_media[key], "span": d_media_span[key]} for key in d_media}
        return tokenzed_conversation, labels, additional_dict

    def __getitem__(self, index):
        try:
            return self.get_item_func(index)
        except Exception as e:
            if not isinstance(e, LabelAllZeroError):
                print(f"Item {index} errored, annotation:\n"
                      f"{self.ann[index]}\n"
                      f"Error:\n"
                      f"{traceback.format_exc()}", force=True)
            for group_name, indices_this_group in self.group_indices.items():
                if indices_this_group[0] <= index <= indices_this_group[-1]:
                    if index == indices_this_group[0]:
                        new_index = indices_this_group[-1]
                    else:
                        new_index = index - 1
                    return self[new_index]

    def groups(self):
        return list(self.group_indices.values())


    @staticmethod
    def collate_func(data):
        input, label, additional_dict = [_[0] for _ in data], [_[1] for _ in data], [_[2] for _ in data]
        input = torch.stack(input, dim=0)
        label = torch.stack(label, dim=0)
        additional_dict = {key: [_[key] for _ in additional_dict] for key in additional_dict[0]}
        # {
        #  "<image>": [{"data": [...], "span": [...]}, {"data": [...], "span": [...]}]},
        #  "<video_frame>": [{"data": [...], "span": [...]}, {"data": [...], "span": [...]}]},
        #  ...
        # }
        return input, label, additional_dict

def find_sublist(a: list, b:list):
    len_a, len_b = len(a), len(b)
    for i in range(len_a - len_b + 1):
        if a[i:i+len_b] == b:
            return i
    return -1