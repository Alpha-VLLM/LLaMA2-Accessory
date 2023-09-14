# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from sentencepiece import SentencePieceProcessor
from transformers import AutoTokenizer
from logging import getLogger
from typing import List
import os


logger = getLogger()


class Tokenizer:
    def __init__(self, model_path: str):
        """
        Create a tokenizer, with inner implementation either spm or HF transformers tokenzier
        :param model_path:
            - when using spm tokenizer, should be path to a sentencepiece model with suffix `.model`
            - when using huggingface transformers tokenizer, should be an HF model repo or a local directory,
              containing tokenizer.json and tokenizer_config.json.
        """
        if model_path.endswith(".model"):  # spm tokenizer
            self.tokenizer_type = "spm"
            # reload tokenizer
            assert os.path.isfile(model_path), model_path
            self.tokenizer = SentencePieceProcessor(model_file=model_path)
            logger.info(f"Reloaded SentencePiece model from {model_path}")

            # BOS / EOS token IDs
            self.n_words: int = self.tokenizer.vocab_size()
            self.bos_id: int = self.tokenizer.bos_id()
            self.eos_id: int = self.tokenizer.eos_id()
            assert self.tokenizer.vocab_size() == self.tokenizer.get_piece_size()
        else:
            self.tokenizer_type = "transformers"
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info(f"load HF transformers tokenizer from {model_path}")
            # BOS / EOS token IDs
            self.n_words: int = self.tokenizer.vocab_size
            self.bos_id: int = self.tokenizer.bos_token_id
            if self.bos_id is None:
                self.bos_id = self.tokenizer.eos_token_id
            self.eos_id: int = self.tokenizer.eos_token_id
            assert self.eos_id is not None

        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        if self.tokenizer_type == "transformers":
            t = self.tokenizer.encode(s, truncation=False, add_special_tokens=False)
        else:
            t = self.tokenizer.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)
