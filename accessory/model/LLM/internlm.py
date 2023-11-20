from typing import Optional, Tuple, Union
from dataclasses import dataclass
import math
import functools

import torch.nn as nn
import torch
import torch.nn.functional as F

from torch.nn.init import normal_

from einops import rearrange

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear
)

from ..components import RMSNorm

from accessory.configs import global_configs
if global_configs.USE_FLASH_ATTENTION:
    from flash_attn import flash_attn_func

from .llama import precompute_freqs_cis, reshape_for_broadcast


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], 2, -1).transpose(-1, -2).contiguous())
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], 2, -1).transpose(-1, -2).contiguous())
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)



@dataclass
class ModelArgs:
    num_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    mlp_ratio: int = 8/3
    drop_rate: float = 0.0
    layer_norm_epsilon: float = 1e-5
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-5
    use_scaled_init: bool = True
    use_swiglu: bool = True
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    rope_theta: float = 10000

    max_batch_size: int = 32
    max_seq_len: int = 2048

    rope_scaling: Optional[float] = None


class MHA(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int = None):
        super().__init__()
        self.layer_idx = layer_idx

        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.num_attention_heads // model_parallel_size
        assert args.hidden_size % args.num_attention_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = args.hidden_size // args.num_attention_heads

        # notice here should change bias=True
        self.Wqkv = ColumnParallelLinear(
            args.hidden_size,
            3 * args.hidden_size,
            bias=True,
            gather_output=False,
        )

        # output projection always have the bias (for now)
        self.out_proj = RowParallelLinear(
            args.hidden_size,
            args.hidden_size,
            bias=True,
            input_is_parallel=True,
        )

        self.args = args

        self.flash = global_configs.USE_FLASH_ATTENTION
        self.k_cache, self.v_cache = None, None

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
        mask: Union[torch.Tensor, str, None]
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, d=self.head_dim)
        xq, xk, xv = qkv.unbind(dim=2)

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
        if use_flash:
            # repeating k/v heads is included in flash_attn
            output = flash_attn_func(xq, keys, values, dropout_p=0.0, causal=is_causal)
            output = output.contiguous().view(bsz, seqlen, -1)
        else:
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
            output = output.transpose(
                1, 2
            ).contiguous().view(bsz, seqlen, -1)

        return self.out_proj(output)

    def allocate_kv_cache(self, max_batch_size: int, max_seq_len: int) -> None:
        kv_cache_shape = (max_batch_size, max_seq_len, self.n_local_heads, self.head_dim)
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
        out_dim: int,
        bias: bool,
        multiple_of: int,
    ):
        super().__init__()

        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=bias, gather_output=False
        )
        self.w2 = RowParallelLinear(
            dim, hidden_dim, bias=bias, input_is_parallel=True
        )
        self.w3 = ColumnParallelLinear(
            hidden_dim, out_dim, bias=bias, gather_output=False
        )

    # @torch.compile
    def _silu_gating(self, x, y):
        return F.silu(x) * y

    def forward(self, x):
        return self.w3(self._silu_gating(self.w1(x), self.w2(x)))


class PackedFlashBaseLayer1D(nn.Module):
    def __init__(self, layer_idx, args:ModelArgs):
        super().__init__()
        self.layer_idx = layer_idx

        self.mixer = MHA(args, layer_idx)

        self.dropout1 = nn.Dropout(args.drop_rate)
        if args.norm_type == "rmsnorm":
            self.norm1 = RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
            self.norm2 = RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        else:
            self.norm1 = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_epsilon)
            self.norm2 = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_epsilon)

        self.mlp = FeedForward(
            args.hidden_size,
            int(args.hidden_size * args.mlp_ratio),
            out_dim=args.hidden_size,
            bias=False,
            multiple_of=args.multiple_of
        )

        self.dropout2 = nn.Dropout(args.drop_rate)
        self.use_swiglu = args.use_swiglu
        self.use_scaled_init = args.use_scaled_init
        # self.residual_in_fp32 = args.residual_in_fp32  # only make sense when using prenorm
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for name, param in self.mixer.named_parameters():
                if param.ndim == 1:
                    param.data.zero_()
                elif "Wqkv" in name:
                    normal_(param.data, std=0.006)
                elif self.use_scaled_init:
                    torch.nn.init.normal_(param.data, mean=0.0, std=0.006 / math.sqrt(2.0 * self.layer_idx + 1))
                else:
                    normal_(param.data, std=0.0015)

            for name, param in self.mlp.named_parameters():
                if param.ndim == 1 and "bias" in name:
                    param.data.zero_()
                elif self.use_swiglu:
                    if self.use_scaled_init and "w2" in name:
                        torch.nn.init.normal_(param.data, mean=0.0, std=0.006 / math.sqrt(2.0 * self.layer_idx + 1))
                    else:
                        normal_(param.data, std=0.006 if "w1" in name or "w2" in name else 0.0015)
                else:
                    if self.use_scaled_init and "fc1" not in name:
                        torch.nn.init.normal_(param.data, mean=0.0, std=0.006 / math.sqrt(2.0 * self.layer_idx + 1))
                    else:
                        normal_(param.data, std=0.006 if "fc1" in name else 0.0015)

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
        mask: Union[torch.Tensor, str, None]
    ) -> torch.Tensor:

        dropped = self.dropout1(x)
        residual = dropped
        hidden_states = self.norm1(residual)

        # if self.residual_in_fp32:
        #     residual = residual.to(torch.float32)
        hidden_states = self.mixer(hidden_states, start_pos, freqs_cis, mask)

        dropped = self.dropout2(hidden_states)
        residual = dropped + residual
        hidden_states = self.norm2(residual)

        # if self.residual_in_fp32:
        #     residual = residual.to(torch.float32)

        hidden_states = self.mlp(hidden_states)

        return hidden_states + residual


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs, with_visual=False):
        super().__init__()
        if with_visual:
            raise NotImplementedError

        self.args = args
        self.vocab_size = args.vocab_size

        self.embedding = ParallelEmbedding(args.vocab_size, args.hidden_size)

        self.layers = nn.ModuleList(
            [
                PackedFlashBaseLayer1D(lid, args)
                for lid in range(args.num_layers)
            ]
        )

        if args.norm_type == "rmsnorm":
            self.norm = RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        else:
            self.norm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.head = ColumnParallelLinear(
            args.hidden_size, args.vocab_size, bias=False
        )
        for _, param in self.head.named_parameters():
            normal_(param, std=0.0052)

        self.freqs_cis = precompute_freqs_cis(
            self.args.hidden_size // self.args.num_attention_heads, self.args.max_seq_len * 2,
            theta=self.args.rope_theta, scaling=self.args.rope_scaling
        )

        self.image_words = 0
        self.cache_image_words = 0 # for inference

    def get_trainable_params(self):
        trainable = {}
        for name, para in self.named_parameters():
            trainable[name] = para

        return trainable


    def forward(self, examples, image=None):
        if image is not None:
            raise NotImplementedError
        self._destroy_kv_cache()  # training always disables kv cache

        _bsz, seqlen = examples.shape
        h = self.embedding(examples)
        self.freqs_cis = self.freqs_cis.to(h.device)

        freqs_cis = self.freqs_cis[:seqlen]
        for _, layer in enumerate(self.layers):
            h = layer(
                h,
                start_pos=0,
                freqs_cis=freqs_cis,
                mask="causal"
            )

        h = self.norm(h)
        h = self.head(h)

        return h

    @torch.inference_mode()
    def forward_inference(self, tokens: torch.Tensor, start_pos: int, image=None):
        _bsz, seqlen = tokens.shape
        if start_pos == 0:
            self._allocate_kv_cache(_bsz)  # kv cache will not re-allocate if size is unchanged
        h = self.embedding(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)

        if image is not None:
            raise NotImplementedError
        else:
            if start_pos == 0:
                self.cache_image_words = 0
                freqs_cis = self.freqs_cis[0: seqlen]
            else:
                # if image was not None when start_pos=0,
                # the offset should be added to start_pos within later forward_inference calls
                start_pos = start_pos + self.cache_image_words
                freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        # Despite that "causal" also works for seqlen == 1, keep it to None for possibly
        # better performance
        mask = None if seqlen == 1 else "causal"

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h)
        output = self.head(h[:, -1, :])  # only compute last logits
        return output

    def _allocate_kv_cache(self, max_batch_size: int) -> None:
        for layer in self.layers:
            layer.mixer.allocate_kv_cache(max_batch_size, self.args.max_seq_len)

    def _destroy_kv_cache(self) -> None:
        for layer in self.layers:
            layer.mixer.destroy_kv_cache()


