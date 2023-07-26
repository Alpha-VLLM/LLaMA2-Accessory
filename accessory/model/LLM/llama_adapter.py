# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional, Tuple
from dataclasses import dataclass
import math
import functools

import torch
from torch import nn
import torch.nn.functional as F

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear
)
from ..peft import LoraColumnParallelLinear, LoraRowParallelLinear

from apex.normalization import FusedRMSNorm as RMSNorm
import open_clip

import configs.global_configs
if configs.global_configs.USE_FLASH_ATTENTION:
    from flash_attn import flash_attn_func

default_linear_init = functools.partial(nn.init.kaiming_uniform_, a=math.sqrt(5))


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

    prefix_layers: Optional[int] = None # prefix, set to n_layers by default
    prefix_len: int = 30

    lora_rank: int = -1 # lora

    bias_tuning: bool = False  # bias


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = LoraColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=args.bias_tuning,
            gather_output=False,
            init_method=default_linear_init,
            lora_rank=args.lora_rank
        )
        self.wk = LoraColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=args.bias_tuning,
            gather_output=False,
            init_method=default_linear_init,
            lora_rank=args.lora_rank
        )
        self.wv = LoraColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=args.bias_tuning,
            gather_output=False,
            init_method=default_linear_init,
            lora_rank=args.lora_rank
        )
        self.wo = LoraRowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=args.bias_tuning,
            input_is_parallel=True,
            init_method=default_linear_init,
            lora_rank=args.lora_rank
        )

        self.args = args

        self.flash = configs.global_configs.USE_FLASH_ATTENTION

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor],
                prefix: Optional[torch.Tensor]=None, prefix_gate: Optional[torch.Tensor]=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = xk
        values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        if prefix is not None:
            prefix_k = self.wk(prefix).repeat(bsz, 1, 1).view(bsz, self.args.prefix_len, self.n_local_heads,
                                                               self.head_dim)
            prefix_v = self.wv(prefix).repeat(bsz, 1, 1).view(bsz, self.args.prefix_len, self.n_local_heads,
                                                               self.head_dim)

        if self.flash:
            output = flash_attn_func(xq, keys, values, dropout_p=0.0, causal=True)

            if prefix is not None:
                prefix_delta = flash_attn_func(xq, prefix_k, prefix_v, dropout_p=0.0, causal=False)
                output = output + prefix_gate.view(1, 1, -1, 1).tanh() * prefix_delta

            output = output.contiguous().view(bsz, seqlen, -1)
        else:
            xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)
            output = F.scaled_dot_product_attention(xq, keys, values, dropout_p=0.0, mask=mask)

            if prefix is not None:
                prefix_k = prefix_k.transpose(1, 2)
                prefix_v = prefix_v.transpose(1, 2)
                prefix_delta = F.scaled_dot_product_attention(xq, prefix_k, prefix_v, dropout_p=0.0, causal=False)
                output = output + prefix_gate.view(1, -1, 1, 1).tanh() * prefix_delta

            output = output.transpose(
                1, 2
            ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        args: ModelArgs,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = LoraColumnParallelLinear(
            dim, hidden_dim, bias=args.bias_tuning, gather_output=False,
            init_method=default_linear_init, lora_rank=args.lora_rank
        )
        self.w2 = LoraRowParallelLinear(
            hidden_dim, dim, bias=args.bias_tuning, input_is_parallel=True,
            init_method=default_linear_init, lora_rank=args.lora_rank
        )
        self.w3 = LoraColumnParallelLinear(
            dim, hidden_dim, bias=args.bias_tuning, gather_output=False,
            init_method=default_linear_init, lora_rank=args.lora_rank
        )

    # @torch.compile
    def _silu_gating(self, x, y):
        return F.silu(x) * y

    def forward(self, x):
        return self.w2(self._silu_gating(self.w1(x), self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            args=args
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def _forward_ffn(self, h):
        return h + self.feed_forward(self.ffn_norm(h))

    def _forward_attention(self, x, start_pos, freqs_cis, mask, prefix, prefix_gate):
        return x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask, prefix, prefix_gate)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor],
                prefix: Optional[torch.Tensor]=None, prefix_gate: Optional[torch.Tensor]=None):
        h = self._forward_attention(x, start_pos, freqs_cis, mask, prefix, prefix_gate)
        out = self._forward_ffn(h)
        return out


class Transformer(nn.Module):
    is_peft = True
    def __init__(self, params: ModelArgs, with_visual=False):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=default_linear_init
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=default_linear_init
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

        self.image_words = 0
        if with_visual:
            print("build llama model with clip")
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
            self.clip, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
            torch.set_default_tensor_type(torch.FloatTensor)
            for name, param in self.clip.named_parameters():
                param.requires_grad = False
            in_dim = self.clip.visual.proj.shape[1]
            # in_dim = 3
            self.clip_proj = nn.Linear(in_dim, params.dim)
            self.clip_proj_norm = nn.LayerNorm(params.dim)
            self.image_words = 257

        # prefix tuning with zero-init attention
        if params.prefix_len > 0:
            prefix_layers = params.prefix_layers if params.prefix_layers is not None else params.n_layers
            self.prefix_layers = prefix_layers
            print(f"create prefix-tuning model with prefix_len {params.prefix_len} and prefix_layers {prefix_layers}")
            n_local_heads = self.layers[0].attention.n_local_heads
            self.prefix_gate = torch.nn.Parameter(torch.zeros(prefix_layers, n_local_heads))
            self.prefix_gate.is_model_parallel = True
            self.prefix = torch.nn.Parameter(torch.zeros(prefix_layers, 1, params.prefix_len, params.dim))
            torch.nn.init.normal_(self.prefix)
        else:
            self.prefix_layers = 0

        self.set_default_trainability()


    def get_trainable_params(self):
        trainable = {}
        for name, para in self.named_parameters():
            if not name.startswith("clip."):
                trainable_key_words = ['norm', 'prefix', 'bias', 'lora']
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


    @torch.no_grad()
    def clip_encode_image(self, x):
        # modified from CLIP
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1,
                      x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # preserve all spatial tokens
        x = self.clip.visual.ln_post(x[:, :, :])

        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj

        return x


    def encode_image(self, image):
        # return self.patch_embed(image)
        image_tokens = self.clip_encode_image(image)
        image_tokens = self.clip_proj_norm(self.clip_proj(image_tokens))
        return image_tokens

    def forward(self, examples, image=None):
        _bsz, seqlen = examples.shape
        h = self.tok_embeddings(examples)
        self.freqs_cis = self.freqs_cis.to(h.device)
        start_pos = 0

        if image is not None:
            image_tokens = self.encode_image(image)
            h = torch.cat((image_tokens, h), dim=1)
            start_pos = image_tokens.shape[1]
            seqlen = h.shape[1]

        # print(f"image: {start_pos}, text: {seqlen - start_pos}, seq_len: {seqlen}")

        freqs_cis = self.freqs_cis[:seqlen]
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=1).type_as(h)

        for layer in self.layers[:-1 * self.prefix_layers]:
            h = layer(h, start_pos, freqs_cis, mask)
        prefix_index = 0
        for layer in self.layers[-1 * self.prefix_layers:]:
            prefix_gate_this_layer = self.prefix_gate[prefix_index]
            prefix_this_layer = self.prefix[prefix_index]
            h = layer(h, start_pos, freqs_cis, mask, prefix_this_layer, prefix_gate_this_layer)
            prefix_index += 1

        h = self.norm(h)
        output = self.output(h[:, start_pos:, :])
        return output


    @torch.inference_mode()
    def forward_inference(self, tokens: torch.Tensor, start_pos: int, image=None):
        assert start_pos==0
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)

        if image is not None:
            image_tokens = self.encode_image(image)
            h = torch.cat((image_tokens, h), dim=1)
            start_pos = start_pos + image_tokens.shape[1]
            seqlen = h.shape[1]

        freqs_cis = self.freqs_cis[:seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1).type_as(h)
            mask[:, :, :, :start_pos] = 0


        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()
