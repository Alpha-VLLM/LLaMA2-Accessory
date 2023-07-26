from dataclasses import dataclass
import functools

from ..peft import wrap_lora

from .llama import (
    ModelArgs as LLaMAModelArgs,
    Transformer as LLaMATransformer,
)

@dataclass
class ModelArgs(LLaMAModelArgs):
    lora_rank: int = -1 # lora
    bias_tuning: bool = True  # bias


class Transformer(LLaMATransformer):
    def __init__(self, args: ModelArgs, with_visual: bool = False) -> None:
        super().__init__(args, with_visual)
        self._setup_peft()
        self.set_default_trainability()

    def _setup_peft(self):
        wrap_lora_with_args = functools.partial(
            wrap_lora,
            lora_rank=self.params.lora_rank,
            bias=self.params.bias_tuning,
        )
        def wrap_attn(attn):
            attn.wq = wrap_lora_with_args(attn.wq)
            attn.wk = wrap_lora_with_args(attn.wk)
            attn.wv = wrap_lora_with_args(attn.wv)
            attn.wo = wrap_lora_with_args(attn.wo)
        def wrap_ffn(ffn):
            ffn.w1 = wrap_lora_with_args(ffn.w1)
            ffn.w2 = wrap_lora_with_args(ffn.w2)
            ffn.w3 = wrap_lora_with_args(ffn.w3)
        for layer in self.layers:
            wrap_attn(layer.attention)
            wrap_ffn(layer.feed_forward)

    def get_trainable_params(self):
        trainable = {}
        for name, para in self.named_parameters():
            if not name.startswith("clip."):
                trainable_key_words = ['norm', 'bias', 'lora']
                if any([_ in name for _ in trainable_key_words]):
                    trainable[name] = para
        return trainable

    def set_default_trainability(self):
        for key, value in self.named_parameters():
            value.requires_grad = False
            value.data = value.data.half()
        for key, value in self.get_trainable_params().items():
            value.data = value.data.float()
            value.requires_grad = True

