# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional, Tuple, Union
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
from util.tensor_type import default_tensor_type

default_linear_init = functools.partial(nn.init.kaiming_uniform_, a=math.sqrt(5))

from .llama import precompute_freqs_cis, reshape_for_broadcast, apply_rotary_emb, repeat_kv


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

    rope_scaling: Optional[float] = None

    prefix_layers: Optional[int] = None # prefix, set to n_layers by default
    prefix_len: int = 30
    v_embed_dim = 768 # latent dim for clip projection layer
    v_depth = 8 # number of perceiver layers for clip projection
    v_num_heads = 16
    v_mlp_ratio = 4.0

    lora_rank: int = -1 # lora

    bias_tuning: bool = False  # bias


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
        self.k_cache, self.v_cache = None, None

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
        mask: Union[torch.Tensor, str, None],
        prefix: Optional[torch.Tensor]=None, prefix_gate: Optional[torch.Tensor]=None
    ) -> torch.Tensor:
        """
        Supported mask spec:
        1. Float tensor: The tensor is added to the attention score matrix.
        2. Boolean tensor: Substitute the ``True`` values with ``0.0`` and ``False`` values with
           ``-inf``, then process in the same way as the float tensor.
        3. str: Currently the only supported choice is ``causal``, for which each token attends
           to all tokens appearing no later than itself. Our implementation assumes the query and
           key sequences aligns on the right for ``causal`` if their lengths are not equal.
        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # if cache is enabled, prepend keys and values in the history.
        if self.k_cache is None or self.v_cache is None:
            keys, values = xk, xv
        else:
            self.k_cache = self.k_cache.to(xk)
            self.v_cache = self.v_cache.to(xv)
            self.k_cache[:bsz, start_pos: start_pos + seqlen, :, :] = xk
            self.v_cache[:bsz, start_pos: start_pos + seqlen, :, :] = xv
            keys = self.k_cache[:bsz, :start_pos + seqlen]
            values = self.v_cache[:bsz, :start_pos + seqlen]

        is_causal = isinstance(mask, str) and mask == "causal"
        # "causal" dispatches to flash_attn only when q and k have the same seqlen
        # because currently the flash_attn causal impl for unequal q & k length is not suited
        # for generation: Generation with cache requires aligning on the right, while the
        # current flash_attn impl aligns on the left. For example, we expect the mask to be
        # as the left one, while the current flash_attn impl gives the right one
        #
        #              K                     K
        #        1 1 1 1 1 0 0         1 0 0 0 0 0 0
        #     Q  1 1 1 1 1 1 0       Q 1 1 0 0 0 0 0
        #        1 1 1 1 1 1 1         1 1 1 0 0 0 0
        use_flash = (
            self.flash  # user configuration
            and (mask is None or (is_causal and keys.size(1) == xq.size(1)))  # supported mask
        )
        if prefix is not None:
            prefix_k = self.wk(prefix).view(bsz, self.args.prefix_len, self.n_local_heads,
                                                               self.head_dim)
            prefix_v = self.wv(prefix).view(bsz, self.args.prefix_len, self.n_local_heads,
                                                               self.head_dim)

        if use_flash:
            # repeating k/v heads is included in flash_attn
            output = flash_attn_func(xq, keys, values, dropout_p=0.0, causal=is_causal)

            if prefix is not None:
                prefix_delta = flash_attn_func(xq, prefix_k, prefix_v, dropout_p=0.0, causal=False)
                output = output + prefix_gate.view(1, 1, -1, 1).tanh() * prefix_delta

            output = output.contiguous().view(bsz, seqlen, -1)
        else:
            # repeat k/v heads if n_kv_heads < n_heads
            keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
            values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

            xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)
            if isinstance(mask, str):
                if is_causal:
                    mask = self._make_causal_mask(xq.size(2), keys.size(2))
                    mask = mask.to(xq.device, non_blocking=True)
                else:
                    raise NotImplementedError()
            output = F.scaled_dot_product_attention(xq, keys, values, dropout_p=0.0, attn_mask=mask)

            if prefix is not None:
                prefix_k = prefix_k.transpose(1, 2)
                prefix_v = prefix_v.transpose(1, 2)
                prefix_delta = F.scaled_dot_product_attention(xq, prefix_k, prefix_v, dropout_p=0.0, causal=False)
                output = output + prefix_gate.view(1, -1, 1, 1).tanh() * prefix_delta

            output = output.transpose(
                1, 2
            ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)

    def allocate_kv_cache(self, max_batch_size: int, max_seq_len: int) -> None:
        kv_cache_shape = (max_batch_size, max_seq_len, self.n_local_kv_heads, self.head_dim)
        if self.k_cache is None or self.k_cache.size() != kv_cache_shape:
            self.k_cache = torch.empty(kv_cache_shape)
        if self.v_cache is None or self.v_cache.size() != kv_cache_shape:
            self.v_cache = torch.empty(kv_cache_shape)

    def destroy_kv_cache(self) -> None:
        self.k_cache, self.v_cache = None, None

    def _make_causal_mask(self, q_len: int, kv_len: int) -> torch.Tensor:
        q_indices = torch.arange(q_len) - q_len
        kv_indices = torch.arange(kv_len) - kv_len
        causal_mask_bool = q_indices.view(-1, 1) >= kv_indices.view(1, -1)
        return causal_mask_bool

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

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        prefix: Optional[torch.Tensor]=None, prefix_gate: Optional[torch.Tensor]=None
    ) -> torch.Tensor:
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
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2, scaling=self.params.rope_scaling
        )

        self.image_words = 0
        if with_visual:
            print("build llama model with clip")
            with default_tensor_type(dtype=torch.half):
                self.clip, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
            for name, param in self.clip.named_parameters():
                param.requires_grad = False
            in_dim = self.clip.visual.proj.shape[1]
            # in_dim = 3
            self.clip_proj = nn.Linear(in_dim, params.dim)
            self.clip_proj_norm = nn.LayerNorm(params.dim)
            self.image_words = 0 # images does not occupy LLM input tokens with llama_adapter

            assert params.prefix_len > 0, "llama_adapter needs prefix if multi modal"
            self.visual_query = torch.nn.Parameter(torch.zeros(params.prefix_len, params.v_embed_dim))
            torch.nn.init.normal_(self.visual_query)
            from timm.models.vision_transformer import Block as ViTBlock
            self.visual_blocks = nn.ModuleList([
                ViTBlock(params.v_embed_dim, params.v_num_heads, params.v_mlp_ratio, qkv_bias=True)
                for _ in range(params.v_depth)])
            self.visual_proj = nn.Linear(params.v_embed_dim, params.dim)
            self.visual_proj_norm = nn.LayerNorm(params.dim)

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

        self.cache_actual_prefix = None


    def get_trainable_params(self):
        trainable = {}
        for name, para in self.named_parameters():
            if not name.startswith("clip."):
                trainable_key_words = ['norm', 'prefix', 'bias', 'lora']
                if any([_ in name for _ in trainable_key_words]):
                    trainable[name] = para

        return trainable


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


    def encode_image(self, imgs):
        clip_feats = self.clip_encode_image(imgs)
        clip_feats = self.clip_proj_norm(self.clip_proj(clip_feats.float()))

        visual_query = self.visual_query.unsqueeze(
            0).repeat(len(imgs), 1, 1)
        visual_query = torch.cat([visual_query, clip_feats], dim=1)
        for block in self.visual_blocks:
            visual_query = block(visual_query)

        visual_query = visual_query[:, :self.query_len, :]
        visual_query = self.visual_proj(visual_query)
        visual_query = self.visual_proj_norm(visual_query)

        return visual_query


    def forward(self, examples, image=None):
        self._destroy_kv_cache()  # training always disables kv cache
        self.cache_actual_prefix = None # training always disables prefix cache
        _bsz, seqlen = examples.shape
        h = self.tok_embeddings(examples)
        self.freqs_cis = self.freqs_cis.to(h.device)

        if image is not None:
            visual_query = self.encode_image(image)
            actual_prefix = self.prefix + visual_query.unsqueeze(0) # [layers, bsz, seq, dim]
        else:
            actual_prefix = self.prefix.repeat(1, _bsz, 1, 1)

        freqs_cis = self.freqs_cis[:seqlen]
        for layer in self.layers[:-1 * self.prefix_layers]:
            h = layer(h, start_pos=0, freqs_cis=freqs_cis, mask="causal")
        prefix_index = 0
        for layer in self.layers[-1 * self.prefix_layers:]:
            prefix_gate_this_layer = self.prefix_gate[prefix_index]
            prefix_this_layer = actual_prefix[prefix_index]
            h = layer(h, start_pos=0, freqs_cis=freqs_cis, mask="causal",
                      prefix=prefix_this_layer, prefix_gate=prefix_gate_this_layer)
            prefix_index += 1

        h = self.norm(h)
        output = self.output(h)
        return output


    @torch.inference_mode()
    def forward_inference(self, tokens: torch.Tensor, start_pos: int, image=None):
        _bsz, seqlen = tokens.shape
        if start_pos == 0:
            self._allocate_kv_cache(_bsz)  # kv cache will not re-allocate if size is unchanged
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)

        if image is not None:
            assert start_pos == 0
            visual_query = self.encode_image(image)
            self.cache_actual_prefix = self.prefix + visual_query.unsqueeze(0)
        elif start_pos == 0:
            self.cache_actual_prefix = self.prefix.repeat(1, _bsz, 1, 1)

        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        # Despite that "causal" also works for seqlen == 1, keep it to None for possibly
        # better performance
        mask = None if seqlen == 1 else "causal"

        for layer in self.layers[:-1 * self.prefix_layers]:
            h = layer(h, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask)
        prefix_index = 0
        for layer in self.layers[-1 * self.prefix_layers:]:
            prefix_gate_this_layer = self.prefix_gate[prefix_index]
            prefix_this_layer = self.cache_actual_prefix[prefix_index]
            h = layer(h, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask,
                      prefix=prefix_this_layer, prefix_gate=prefix_gate_this_layer)
            prefix_index += 1

        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output

    def _allocate_kv_cache(self, max_batch_size: int) -> None:
        for layer in self.layers:
            layer.attention.allocate_kv_cache(max_batch_size, self.params.max_seq_len)

    def _destroy_kv_cache(self) -> None:
        for layer in self.layers:
            layer.attention.destroy_kv_cache()
