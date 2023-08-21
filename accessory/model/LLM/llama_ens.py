# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional, Tuple, Union
from dataclasses import dataclass
import math
import functools

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear
)

from apex.normalization import FusedRMSNorm as RMSNorm
from transformers import Blip2Processor, Blip2Model
import open_clip


import configs.global_configs
if configs.global_configs.USE_FLASH_ATTENTION:
    from flash_attn import flash_attn_func

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


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=default_linear_init,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=default_linear_init,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=default_linear_init,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=default_linear_init,
        )

        self.args = args

        self.flash = configs.global_configs.USE_FLASH_ATTENTION
        self.k_cache, self.v_cache = None, None

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
        mask: Union[torch.Tensor, str, None]
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
        if use_flash:
            # repeating k/v heads is included in flash_attn
            output = flash_attn_func(xq, keys, values, dropout_p=0.0, causal=is_causal)
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
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=default_linear_init
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=default_linear_init
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=default_linear_init
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
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def _forward_ffn(self, h):
        return h + self.feed_forward(self.ffn_norm(h))

    def _forward_attention(self, x, start_pos, freqs_cis, mask):
        return x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
        mask: Union[torch.Tensor, str, None]
    ) -> torch.Tensor:
        h = self._forward_attention(x, start_pos, freqs_cis, mask)
        out = self._forward_ffn(h)
        return out


class Transformer(nn.Module):
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
        self.cache_image_words = 0 # for inference
        if with_visual:
            print("build llama model with qformerv2")
            self.qformer = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

            self.qformer.language_projection = None
            self.qformer.language_model = None

            self.qformer_proj = nn.Sequential(
                nn.Linear(768, params.dim),
                nn.LayerNorm(params.dim)
            )

            print("build llama model with clip")
            self.clip, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
            self.clip.transformer = None

            print("build llama model with openclip")
            self.openclip_convnext_xxl, _, _ = open_clip.create_model_and_transforms(
                "convnext_xxlarge", pretrained="laion2b_s34b_b82k_augreg_soup"
            )
            self.openclip_convnext_xxl = self.openclip_convnext_xxl.visual.trunk
            self.openclip_convnext_xxl.head.global_pool = nn.Identity()
            self.openclip_convnext_xxl.head.flatten = nn.Identity()

            print("build llama model with dinov2")
            import os.path
            if os.path.exists("/mnt/petrelfs/gaopeng/.cache/torch/hub/facebookresearch_dinov2_main"):
                self.dinov2_vitg14 = torch.hub.load("/mnt/petrelfs/gaopeng/.cache/torch/hub/facebookresearch_dinov2_main", "dinov2_vitg14", source="local")
            else:
                self.dinov2_vitg14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14")

            self.visual_proj = nn.Sequential(
                nn.Linear(3072 + 1024 + 1536, params.dim),
                nn.LayerNorm(params.dim),
            )

            self.image_words = 30 + 257 + 2
            # add image tags
            self.start_img = nn.Parameter(torch.rand(1, 1, params.dim))
            self.end_img = nn.Parameter(torch.rand(1, 1, params.dim))


    def get_trainable_params(self):
        trainable = {}
        no_train_prefix = ["qformer.", "openclip_convnext_xxl.", "clip.", "dinov2_vitg14."]
        for name, para in self.named_parameters():
            if not any([name.startswith(_) for _ in no_train_prefix]):
                trainable[name] = para
            else:
                print(f'not trainable {name}')

        return trainable

    @torch.no_grad()
    def clip_encode_image(self, x):
        # modified from CLIP
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1,
                                                                                  x.shape[-1], dtype=x.dtype,
                                                                                  device=x.device), x],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # preserve all spatial tokens
        x = self.clip.visual.ln_post(x[:, :, :])

        # if self.clip.visual.proj is not None:
        #    x = x @ self.clip.visual.proj

        return x

    def encode_image(self, image):
        # [B, 32, 768]
        self.clip.eval()
        self.openclip_convnext_xxl.eval()
        image = image.half()
        image_bs = image.size(0)
        mp_world_size = fs_init.get_model_parallel_world_size()
        mp_rank = fs_init.get_model_parallel_rank()
        # assert image_bs % mp_world_size == 0

        n_pad_items = (mp_world_size - image_bs % mp_world_size) % mp_world_size
        padded_image = torch.cat([image, image[:1].expand(n_pad_items, *image.size()[1:])], dim=0)
        padded_image_bs = padded_image.shape[0]

        local_image_bs = padded_image_bs // mp_world_size
        local_image = padded_image[local_image_bs * mp_rank: local_image_bs * (mp_rank + 1)]
        with torch.no_grad():
            local_image_feats = self.qformer.get_qformer_features(pixel_values=local_image).last_hidden_state
            image_feats = torch.zeros([padded_image_bs, *local_image_feats.size()[1:]],
                                      device=local_image_feats.device, dtype=local_image_feats.dtype)
            dist.all_gather_into_tensor(image_feats, local_image_feats, group=fs_init.get_model_parallel_group())

            local_clip_image_feats = self.clip_encode_image(local_image)
            local_convnext_image_feats = self.openclip_convnext_xxl(
                F.interpolate(local_image, size=(256, 256))
            )
            assert local_convnext_image_feats.size()[1:] == (3072, 8, 8)
            local_convnext_image_feats = local_convnext_image_feats.repeat_interleave(
                2, dim=-1
            ).repeat_interleave(
                2, dim=-2
            )  # (3072, 16, 16)
            local_convnext_image_feats = local_convnext_image_feats.flatten(-2).permute(0, 2, 1)  # (*, 256, 3072)
            local_convnext_image_feats = torch.cat([
                local_convnext_image_feats.mean(dim=1, keepdim=True),  # add gap as cls token
                local_convnext_image_feats,
            ], dim=1)  # (*, 257, 3072)

            clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).to(local_image, non_blocking=True).view(3,
                                                                                                                  1,
                                                                                                                  1)
            clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).to(local_image, non_blocking=True).view(3,
                                                                                                                  1,
                                                                                                                  1)
            dinov2_mean = torch.Tensor([0.485, 0.456, 0.406]).to(local_image, non_blocking=True).view(3, 1, 1)
            dinov2_std = torch.Tensor([0.229, 0.224, 0.225]).to(local_image, non_blocking=True).view(3, 1, 1)
            local_dinov2_image_feats = self.dinov2_vitg14.forward_features(
                (local_image * clip_std + clip_mean - dinov2_mean) / dinov2_std
            )
            local_dinov2_image_feats = torch.cat([
                local_dinov2_image_feats["x_norm_clstoken"].unsqueeze(1),
                local_dinov2_image_feats["x_norm_patchtokens"],
            ], dim=1)
            local_ens_image_feats = torch.cat([
                local_clip_image_feats,
                local_convnext_image_feats,
                local_dinov2_image_feats,
            ], dim=2)  # (*, 257, 5632)

            ens_image_feats = torch.zeros([padded_image_bs, *local_ens_image_feats.size()[1:]],
                                          device=local_ens_image_feats.device, dtype=local_ens_image_feats.dtype)
            dist.all_gather_into_tensor(ens_image_feats, local_ens_image_feats,
                                        group=fs_init.get_model_parallel_group())

            ens_image_feats = ens_image_feats.float()[:image_bs]
            image_feats = image_feats.float()[:image_bs]

        image_feats = self.qformer_proj(image_feats)
        ens_image_feats = self.visual_proj(ens_image_feats)
        image_feats = torch.cat([image_feats, ens_image_feats], dim=1)
        # image_feats = torch.zeros([image.size(0), 32, 768], dtype=torch.half, device=image.device)
        # image_feats = self.qformer_proj(image_feats)

        return image_feats

    def forward(self, examples, image=None):
        self._destroy_kv_cache()  # training always disables kv cache
        _bsz, seqlen = examples.shape
        h = self.tok_embeddings(examples)
        self.freqs_cis = self.freqs_cis.to(h.device)

        image_words = 0
        if image is not None:
            h_bos, h_caption = h[:, :1], h[:, 1:]
            image_tokens = self.encode_image(image)
            h = torch.cat((h_bos, self.start_img.expand(_bsz, -1, -1), image_tokens, self.end_img.expand(_bsz, -1, -1), h_caption), dim=1)
            image_words = image_tokens.shape[1] + 1 + 1
            seqlen = h.shape[1]

        freqs_cis = self.freqs_cis[:seqlen]
        for layer in self.layers:
            h = layer(h, start_pos=0, freqs_cis=freqs_cis, mask="causal")
        h = self.norm(h)
        output = self.output(h[:, image_words:, :])
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
            h_bos, h_caption = h[:, :1], h[:, 1:]
            image_tokens = self.encode_image(image)
            self.cache_image_words = image_tokens.shape[1] + 1 + 1
            h = torch.cat((h_bos, self.start_img.repeat(_bsz, 1, 1), image_tokens, self.end_img.repeat(_bsz, 1, 1), h_caption), dim=1)
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
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()

    def _allocate_kv_cache(self, max_batch_size: int) -> None:
        for layer in self.layers:
            layer.attention.allocate_kv_cache(max_batch_size, self.params.max_seq_len)

    def _destroy_kv_cache(self) -> None:
        for layer in self.layers:
            layer.attention.destroy_kv_cache()


