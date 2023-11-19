from typing import Optional, Tuple, Union
from dataclasses import dataclass

import torch.nn as nn
import torch
import torch.nn.functional as F

from torch.nn import LayerNorm

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear
)

from accessory.configs import global_configs
if global_configs.USE_FLASH_ATTENTION:
    from flash_attn import flash_attn_func

from .llama import precompute_freqs_cis, reshape_for_broadcast, repeat_kv

@dataclass
class ModelArgs:
    num_layers: int = 80
    hidden_size: int = 14848
    num_attention_heads: int = 232

    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    bias: bool = False
    multi_query: bool = True
    new_decoder_architecture: bool = True
    parallel_attn: bool = True
    initializer_range = 0.02
    num_kv_heads: Optional[int] = None
    layer_norm_epsilon: float = 1e-5
    vocab_size: int = -1
    rope_theta: float = 10000

    max_batch_size: int = 32
    max_seq_len: int = 2048

    rope_scaling: Optional[float] = None


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



def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out


class FalconAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int = None):
        super().__init__()
        self.layer_idx = layer_idx

        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.num_attention_heads // model_parallel_size
        assert args.hidden_size % args.num_attention_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = args.hidden_size // args.num_attention_heads
        assert args.num_attention_heads % model_parallel_size == 0

        self.wq = ColumnParallelLinear(
            args.hidden_size,
            args.hidden_size,
            bias=args.bias,
            gather_output=False
        )

        if args.new_decoder_architecture:
            assert args.num_kv_heads % model_parallel_size == 0
            self.wk = ColumnParallelLinear(
                args.hidden_size,
                self.head_dim * args.num_kv_heads,
                bias=args.bias,
                gather_output=False
            )
            self.wv = ColumnParallelLinear(
                args.hidden_size,
                self.head_dim * args.num_kv_heads,
                bias=args.bias,
                gather_output=False
            )
            self.n_local_kv_heads = args.num_kv_heads // model_parallel_size
        if args.multi_query:
            self.wk = nn.Linear(args.hidden_size, self.head_dim, bias=args.bias)
            self.wv = nn.Linear(args.hidden_size, self.head_dim, bias=args.bias)
            self.n_local_kv_heads = 1
        else:
            self.wk = ColumnParallelLinear(
                args.hidden_size,
                args.hidden_size,
                bias=args.bias,
                gather_output=False
            )
            self.wv = ColumnParallelLinear(
                args.hidden_size,
                args.hidden_size,
                bias=args.bias,
                gather_output=False
            )
            self.n_local_kv_heads = self.n_local_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.dense = RowParallelLinear(args.hidden_size, args.hidden_size, bias=args.bias, input_is_parallel=True)

        self.args = args

        self.flash = global_configs.USE_FLASH_ATTENTION
        self.k_cache, self.v_cache = None, None

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
        mask: Union[torch.Tensor, str, None]
    ) -> torch.Tensor:
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
        if use_flash:
            # repeating k/v heads is included in flash_attn
            output = flash_attn_func(xq, keys, values, dropout_p=self.args.attention_dropout, causal=is_causal)
            output = output.contiguous().view(bsz, seqlen, -1)
        else:
            # repeat k/v heads if n_kv_heads < n_heads
            keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim) # todo n_rep
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
            output = F.scaled_dot_product_attention(xq, keys, values,
                                                    dropout_p=self.args.attention_dropout, attn_mask=mask)
            output = output.transpose(
                1, 2
            ).contiguous().view(bsz, seqlen, -1)

        return self.dense(output)

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

class FalconMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.dense_h_to_4h = ColumnParallelLinear(args.hidden_size, 4 * args.hidden_size, bias=args.bias, gather_output=False)
        self.act = nn.GELU()
        self.dense_4h_to_h = RowParallelLinear(4 * args.hidden_size, args.hidden_size, bias=args.bias, input_is_parallel=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.dense_h_to_4h(x))
        x = self.dense_4h_to_h(x)
        return x



class FalconDecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.self_attention = FalconAttention(args)
        self.mlp = FalconMLP(args)
        self.args = args
        self.layer_id = layer_id

        if args.new_decoder_architecture:
            # The layer norm before self-attention
            self.ln_attn = LayerNorm(args.hidden_size, eps=args.layer_norm_epsilon)
            # The layer norm before the MLP
            self.ln_mlp = LayerNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        else:
            self.input_layernorm = LayerNorm(args.hidden_size, eps=args.layer_norm_epsilon)
            if not args.parallel_attn:
                self.post_attention_layernorm = LayerNorm(args.hidden_size, eps=args.layer_norm_epsilon)

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
        mask: Union[torch.Tensor, str, None]
    ) -> torch.Tensor:
        residual = x

        if self.args.new_decoder_architecture:
            attention_layernorm_out = self.ln_attn(x)
            mlp_layernorm_out = self.ln_mlp(x)
        else:
            attention_layernorm_out = self.input_layernorm(x)

        # Self attention.
        attn_output = self.self_attention(
            attention_layernorm_out, start_pos, freqs_cis,
            mask=mask
        )

        if not self.args.new_decoder_architecture:
            if self.args.parallel_attn:
                mlp_layernorm_out = attention_layernorm_out
            else:
                residual = dropout_add(
                    attn_output, residual, self.args.attention_dropout, training=self.training
                )
                mlp_layernorm_out = self.post_attention_layernorm(residual)


        # MLP.
        mlp_output = self.mlp(mlp_layernorm_out)

        if self.args.new_decoder_architecture or self.args.parallel_attn:
            mlp_output += attn_output

        output = dropout_add(mlp_output, residual, self.args.hidden_dropout, training=self.training)

        return output

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs, with_visual=False):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.num_layers
        self.word_embeddings = ParallelEmbedding(
            args.vocab_size, args.hidden_size
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.num_layers):
            self.layers.append(FalconDecoderLayer(layer_id, args))

        self.ln_f = LayerNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.output = ColumnParallelLinear(
            args.hidden_size, args.vocab_size, bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
            self.args.hidden_size // self.args.num_attention_heads, self.args.max_seq_len * 2,
            theta=self.args.rope_theta, scaling=self.args.rope_scaling
        )

        self.image_words = 0
        self.cache_image_words = 0 # for inference
        if with_visual:
            raise NotImplementedError()

    def get_trainable_params(self):
        trainable = {}
        for name, para in self.named_parameters():
            if not name.startswith("clip."):
                trainable[name] = para

        return trainable


    def forward(self, examples, image=None):
        self._destroy_kv_cache()  # training always disables kv cache
        _bsz, seqlen = examples.shape
        h = self.word_embeddings(examples)
        self.freqs_cis = self.freqs_cis.to(h.device)

        image_words = 0
        if image is not None:
            image_tokens = self.encode_image(image)
            image_words = image_tokens.shape[1]
            h = torch.cat((image_tokens, h), dim=1)
            seqlen = h.shape[1]

        freqs_cis = self.freqs_cis[:seqlen]
        for layer in self.layers:
            h = layer(h, start_pos=0, freqs_cis=freqs_cis, mask="causal")
        h = self.ln_f(h)
        output = self.output(h[:, image_words:, :])
        return output


    @torch.inference_mode()
    def forward_inference(self, tokens: torch.Tensor, start_pos: int, image=None):
        _bsz, seqlen = tokens.shape
        if start_pos == 0:
            self._allocate_kv_cache(_bsz)  # kv cache will not re-allocate if size is unchanged
        h = self.word_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)

        if image is not None:
            assert start_pos == 0
            image_tokens = self.encode_image(image)
            self.cache_image_words = image_tokens.shape[1]
            h = torch.cat((image_tokens, h), dim=1)
            seqlen = h.shape[1]
            freqs_cis = self.freqs_cis[0: seqlen]
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
        h = self.ln_f(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()

    def _allocate_kv_cache(self, max_batch_size: int) -> None:
        for layer in self.layers:
            layer.self_attention.allocate_kv_cache(max_batch_size, self.args.max_seq_len)

    def _destroy_kv_cache(self) -> None:
        for layer in self.layers:
            layer.self_attention.destroy_kv_cache()


