import copy
import warnings

from typing import List, Dict
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


class DialogGenerator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.header = f"{conversation_lib.default_conversation.system}"
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
            self.space_part_of_next_word = False
        else:
            sentence3 = self.tokenizer.encode(" " + probe,
                                              bos=False, eos=False)
            assert sentence1[-len(sentence3):] == sentence3
            self.space_part_of_next_word = True

    def add_speaker_and_signal(self, source):
        """Add speaker and start/end signal on each round."""
        sep = "###"
        conversation = self.header + "\n\n" + sep

        to_predict_list = []

        for i, sentence in enumerate(source):
            from_str = sentence["from"]
            if from_str.lower() in ["human"]:
                from_str = conversation_lib.default_conversation.roles[0]
            elif from_str.lower() in ["gpt", "assistant"]:
                from_str = conversation_lib.default_conversation.roles[1]
            else:
                raise ValueError(f"unknown dialog role: {from_str.lower()}")

            value = sentence["value"]
            if value is not None:
                sentence_value = " " + from_str + ": " + value + "\n" + sep
                conversation = conversation + sentence_value

                if from_str == conversation_lib.default_conversation.roles[1]:
                    to_predict_value = value + "\n" + sep
                    if self.space_part_of_next_word:
                        to_predict_value = " " + to_predict_value
                    to_predict_list.append(to_predict_value)
            else:  # for inference, the last turn should be {"from": "gpt", value: None}
                assert i == len(source) - 1
                response_intro = " " + from_str + ":"
                if not self.space_part_of_next_word:
                    response_intro = response_intro + " "
                conversation = conversation + response_intro

        return conversation, to_predict_list


class DialogProcessor:
    def __init__(self, transform, max_words, media_words: Dict, tokenizer: str | Tokenizer):
        self.transform = transform
        print(f"transform:\n{self.transform}")
        self.max_words = max_words
        if isinstance(tokenizer, str):
            self.tokenizer = Tokenizer(model_path=tokenizer)
        else:
            self.tokenizer = copy.deepcopy(tokenizer)
        self.conversation_generator = DialogGenerator(self.tokenizer)

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

        self.implicit_after_bos=True

    def collect_and_process_media(self, data_item):
        """
        this function receives a raw piece of data (e.g. read from `.json` data file),
        and returns d_media, containing the prepared media readily usable by model
        YOU MAY OVERRIDE THIS FUNCTION TO SUPPORT COMPLEX LOADING OF VARIOUS FORMS OF DATA
        """
        d_media = {}
        for media_symbol in self.d_media_symbol_words:
            if media_symbol in data_item:
                l_media = data_item[media_symbol]  # a list of media paths
            elif media_symbol.lstrip("<").rstrip(">") in data_item:
                l_media = data_item[media_symbol.lstrip("<").rstrip(">")]
            else:
                l_media = []
            if not isinstance(l_media, list):  # data with only one media, in format {"image": image_name, ...}
                l_media = [l_media]

            d_media[media_symbol] = []
            for media in l_media:
                if isinstance(media, Image.Image):
                    image = self.transform(media)
                    d_media[media_symbol].append(image)
                else:
                    image = read_img_general(media)
                    image = self.transform(image)
                    d_media[media_symbol].append(image)

        return d_media

    def reserve_space_for_media(self, tokens: List[int], labels):
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

    def insert_implicit_media_token_in_q1(self, conv, d_media):
        """
        Add the media tokens to the beginning of the first instruction from
        human. This logic may be more reasonable. However, it is incompatible
        with old-version Accessory models, which are trained with image tokens
        inserted directly behind the first token (<bos>).
        :param conv: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}, ...]
        :param d_media: a dict of media for all media types
        """
        conv = copy.deepcopy(conv)

        for media_symbol, l_media in d_media.items():
            media_symbol_count = "".join([_["value"] for _ in conv]).count(media_symbol)
            if media_symbol_count > 0:
                assert media_symbol_count == len(l_media), \
                f"{media_symbol_count} {media_symbol} exists in text, but {len(l_media)} actual media are given"
            else:
                conv[0]['value'] = (media_symbol+" ") * len(l_media) + conv[0]['value']

        return conv

    def insert_implicit_meida_token_after_bos(self, tokens, labels, d_media):
        """
        Legacy versions of LLaMA2-Accessory handled media in a non-interleaved
        manner, where image tokens are inserted directly behind the first token,
        namely <bos>. To support interleaved media comprehension and generation,
        Accessory now supports the explicit specification of media occurance,
        which is achieved by adding media symbols, e.g. <image>, within the
        conversations. On the other hand, for media without explicit speicification,
        this function realizes the legacy behavior to arrange them after <bos>.
        :param tokens: tokenized input
        :param labels: tokenized labels
        :param d_media: a dict of media for all media types, for determining how
        many media tokens need to be inserted
        """
        tokens = copy.deepcopy(tokens)
        labels = copy.deepcopy(labels)
        for media_symbol, l_media in d_media.items():
            media_token = self.d_media_symbol_token[media_symbol]
            media_token_count = tokens.count(media_token)

            if media_token_count > 0:
                # if media symbols already exist in dialog data
                # the number of symbols should equal to number of media
                assert media_token_count == len(l_media), \
                    f"{media_token_count} {media_symbol} exists in text, but {len(l_media)} actual media are given"
            else:
                # add media symbols after BOS token
                tokens = tokens[:1] + [media_token] * len(l_media) + tokens[1:]
                labels = labels[:1] + [IGNORE_INDEX] * len(l_media) + labels[1:]
        return tokens, labels


    def process_item(self, data_item, pad=False):
        d_media = self.collect_and_process_media(data_item)

        source = data_item["conversations"]

        if not self.implicit_after_bos:
            source = self.insert_implicit_media_token_in_q1(source, d_media)

        conversation, to_predict_values = self.conversation_generator.add_speaker_and_signal(source)
        # if len(to_predict_values) == 0:
        #     raise ValueError(f"see dialog data with nothing to predict, data: {data_item}")

        print(conversation)
        # dialog does not need eos
        tokens = self.tokenizer.encode(conversation, bos=True, eos=False)
        labels = [IGNORE_INDEX for _ in tokens]

        check_pos = 0
        for value in to_predict_values:
            tokenized_value = self.tokenizer.encode(value, bos=False, eos=False)
            value_pos = find_sublist(tokens[check_pos:], tokenized_value) + check_pos
            if value_pos == -1:
                raise ValueError("a sentence mismatches the corresponding piece in the conversation, data: {data_item}")
            labels[value_pos:value_pos+len(tokenized_value)] = tokenized_value
            assert labels[value_pos:value_pos+len(tokenized_value)] == tokens[value_pos:value_pos+len(tokenized_value)]
            check_pos = value_pos+len(tokenized_value)

        if self.implicit_after_bos:
            tokens, labels = self.insert_implicit_meida_token_after_bos(tokens, labels, d_media)

        tokens, labels, d_media_span = self.reserve_space_for_media(tokens, labels)

        tokens = torch.tensor(tokens, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)

        padding = self.max_words - tokens.shape[0]
        if padding > 0 and pad:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            labels = torch.cat((labels, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_words]
            labels = labels[:self.max_words]
            # avoid truncation within an image span, especially for supporting image generation

            for symbol, l_span in d_media_span.items():
                new_l_span = []
                new_l_media = []
                for span, media in zip(l_span, d_media[symbol]):
                    if span[1] <= self.max_words:
                        new_l_span.append(span)
                        new_l_media.append(media)
                    else:
                        tokens[span[0]:] = -1
                        labels[span[0]:] = -1
                d_media_span[symbol] = new_l_span
                d_media[symbol] = new_l_media
            # warnings.warn(f'Warning for truncation input!\n{data_item}')

        tokens[tokens.lt(0)] = 0
        labels[labels.lt(0)] = 0

        assert len(tokens) == len(labels)

        # if torch.count_nonzero(labels) == 0:
        #     raise LabelAllZeroError()

        additional_dict = {key: {"data": d_media[key], "span": d_media_span[key]} for key in d_media}

        return tokens, labels, additional_dict


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

    def __len__(self):
        return len(self.ann)

    def get_item_func(self, index):
        data_item: Dict = self.ann[index]

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