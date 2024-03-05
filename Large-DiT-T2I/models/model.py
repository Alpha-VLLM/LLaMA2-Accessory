# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import functools
import math
from typing import Optional, Tuple, List

from apex.normalization import FusedRMSNorm as RMSNorm
import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear, RowParallelLinear, ParallelEmbedding,
)
from flash_attn import flash_attn_func
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#############################################################################
#             Embedding Layers for Timesteps and Class Labels               #
#############################################################################

class ParallelTimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            ColumnParallelLinear(
                frequency_embedding_size, hidden_size, bias=True,
                gather_output=False,
                init_method=functools.partial(nn.init.normal_, std=0.02),
            ),
            nn.SiLU(),
            RowParallelLinear(
                hidden_size, hidden_size, bias=True, input_is_parallel=True,
                init_method=functools.partial(nn.init.normal_, std=0.02),
            ),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32
            ) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([
                embedding, torch.zeros_like(embedding[:, :1])
            ], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb


class ParallelLabelEmbedder(nn.Module):
    r"""Embeds class labels into vector representations. Also handles label
    dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = int(dropout_prob > 0)
        self.embedding_table = ParallelEmbedding(
            num_classes + use_cfg_embedding, hidden_size,
            init_method=functools.partial(nn.init.normal_, std=0.02),
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(
                labels.shape[0], device=labels.device
            ) < self.dropout_prob
            drop_ids = drop_ids.cuda()
            dist.broadcast(
                drop_ids,
                fs_init.get_model_parallel_src_rank(),
                fs_init.get_model_parallel_group(),
            )
            drop_ids = drop_ids.to(labels.device)
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#############################################################################
#                               Core DiT Model                              #
#############################################################################


class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, dim: int, n_heads: int, n_kv_heads: Optional[int], qk_norm: bool, y_dim: int):
        """
        Initialize the Attention module.

        Args:
            dim (int): Number of input dimensions.
            n_heads (int): Number of heads.
            n_kv_heads (Optional[int]): Number of kv heads, if using GQA.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        self.wq = ColumnParallelLinear(
            dim, n_heads * self.head_dim, bias=False, gather_output=False,
            init_method=nn.init.xavier_uniform_,
        )
        self.wk = ColumnParallelLinear(
            dim, self.n_kv_heads * self.head_dim, bias=False,
            gather_output=False, init_method=nn.init.xavier_uniform_,
        )
        self.wv = ColumnParallelLinear(
            dim, self.n_kv_heads * self.head_dim, bias=False,
            gather_output=False, init_method=nn.init.xavier_uniform_,
        )
        if y_dim > 0:
            self.wk_y = ColumnParallelLinear(
                y_dim, self.n_kv_heads * self.head_dim, bias=False,
                gather_output=False, init_method=nn.init.xavier_uniform_,
            )
            self.wv_y = ColumnParallelLinear(
                y_dim, self.n_kv_heads * self.head_dim, bias=False,
                gather_output=False, init_method=nn.init.xavier_uniform_,
            )
            self.gate = nn.Parameter(torch.zeros([self.n_local_heads]))

        self.wo = RowParallelLinear(
            n_heads * self.head_dim, dim, bias=False,
            input_is_parallel=True, init_method=nn.init.xavier_uniform_,
        )

        if qk_norm:
            self.q_norm = nn.LayerNorm(self.n_local_heads * self.head_dim)
            self.k_norm = nn.LayerNorm(self.n_local_kv_heads * self.head_dim)
            if y_dim > 0:
                self.ky_norm = nn.LayerNorm(self.n_local_kv_heads * self.head_dim)
            else:
                self.ky_norm = nn.Identity()
        else:
            self.q_norm = self.k_norm = nn.Identity()
            self.ky_norm = nn.Identity()

    @staticmethod
    def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
        """
        Reshape frequency tensor for broadcasting it with another tensor.

        This function reshapes the frequency tensor to have the same shape as
        the target tensor 'x' for the purpose of broadcasting the frequency
        tensor during element-wise operations.

        Args:
            freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
            x (torch.Tensor): Target tensor for broadcasting compatibility.

        Returns:
            torch.Tensor: Reshaped frequency tensor.

        Raises:
            AssertionError: If the frequency tensor doesn't match the expected
                shape.
            AssertionError: If the target tensor 'x' doesn't have the expected
                number of dimensions.
        """
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1
                 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to input tensors using the given frequency
        tensor.

        This function applies rotary embeddings to the given query 'xq' and
        key 'xk' tensors using the provided frequency tensor 'freqs_cis'. The
        input tensors are reshaped as complex numbers, and the frequency tensor
        is reshaped for broadcasting compatibility. The resulting tensors
        contain rotary embeddings and are returned as real tensors.

        Args:
            xq (torch.Tensor): Query tensor to apply rotary embeddings.
            xk (torch.Tensor): Key tensor to apply rotary embeddings.
            freqs_cis (torch.Tensor): Precomputed frequency tensor for complex
                exponentials.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor
                and key tensor with rotary embeddings.
        """
        with torch.cuda.amp.autocast(enabled=False):
            xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
            xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
            freqs_cis = Attention.reshape_for_broadcast(freqs_cis, xq_)
            xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
            xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
            return xq_out.type_as(xq), xk_out.type_as(xk)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, y: torch.Tensor, y_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        dtype = xq.dtype

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = Attention.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xq, xk = xq.to(dtype), xk.to(dtype)

        if dtype in [torch.float16, torch.bfloat16]:
            output = flash_attn_func(xq, xk, xv, dropout_p=0., causal=False)
        else:
            n_rep = self.n_local_heads // self.n_local_kv_heads
            if n_rep >= 1:
                xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
                xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            output = F.scaled_dot_product_attention(
                xq.permute(0, 2, 1, 3),
                xk.permute(0, 2, 1, 3),
                xv.permute(0, 2, 1, 3),
                dropout_p=0., is_causal=False,
            ).permute(0, 2, 1, 3)
        output = output.flatten(-2)

        if hasattr(self, "wk_y"):
            yk = self.ky_norm(self.wk_y(y)).view(bsz, -1, self.n_local_kv_heads, self.head_dim)
            yv = self.wv_y(y).view(bsz, -1, self.n_local_kv_heads, self.head_dim)
            n_rep = self.n_local_heads // self.n_local_kv_heads
            if n_rep >= 1:
                yk = yk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
                yv = yv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            output_y = F.scaled_dot_product_attention(
                xq.permute(0, 2, 1, 3),
                yk.permute(0, 2, 1, 3),
                yv.permute(0, 2, 1, 3),
                y_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seqlen, -1)
            ).permute(0, 2, 1, 3)
            output_y = output_y * self.gate.tanh().view(1, 1, -1, 1)
            output_y = output_y.flatten(-2)
            output = output + output_y

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple
                of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden
                dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first
                layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third
                layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * (
            (hidden_dim + multiple_of - 1) // multiple_of
        )

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False,
            init_method=nn.init.xavier_uniform_,
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True,
            init_method=nn.init.xavier_uniform_,
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False,
            init_method=nn.init.xavier_uniform_,
        )

    @torch.compile
    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, dim: int, n_heads: int, n_kv_heads: int,
                 multiple_of: int, ffn_dim_multiplier: float, norm_eps: float,
                 qk_norm: bool, y_dim: int) -> None:
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            dim (int): Embedding dimension of the input features.
            n_heads (int): Number of attention heads.
            n_kv_heads (Optional[int]): Number of attention heads in key and
                value features (if using GQA), or set to None for the same as
                query.
            multiple_of (int): Value to ensure hidden dimension is a multiple
                of this value in the FeedForward block.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden
                dimension in the FeedForward block. Defaults to None.
            norm_eps (float): A small value added to the norm layer
                denominators to avoid division-by-zero.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(dim, n_heads, n_kv_heads, qk_norm, y_dim)
        self.feed_forward = FeedForward(
            dim=dim, hidden_dim=4 * dim, multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            ColumnParallelLinear(
                min(dim, 1024), 6 * dim, bias=True, gather_output=True,
                init_method=nn.init.zeros_,
            ),
        )

        self.attention_y_norm = RMSNorm(y_dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention.
                Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and
                feedforward layers.

        """
        if adaln_input is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                self.adaLN_modulation(adaln_input).chunk(6, dim=1)

            x = x + gate_msa.unsqueeze(1) * self.attention(
                modulate(self.attention_norm(x), shift_msa, scale_msa),
                freqs_cis,
                self.attention_y_norm(y), y_mask
            )
            x = x + gate_mlp.unsqueeze(1) * self.feed_forward(
                modulate(self.ffn_norm(x), shift_mlp, scale_mlp),
            )

        else:
            x = x + self.attention(
                self.attention_norm(x), freqs_cis, self.attention_y_norm(y), y_mask,
            )
            x = x + self.feed_forward(self.ffn_norm(x))

        return x

class ParallelFinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6,
        )
        self.linear = ColumnParallelLinear(
            hidden_size, patch_size * patch_size * out_channels, bias=True,
            init_method=nn.init.zeros_, gather_output=True,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            ColumnParallelLinear(
                min(hidden_size, 1024), 2 * hidden_size, bias=True,
                init_method=nn.init.zeros_, gather_output=True,
            ),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT_Llama(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        patch_size: int = 2,
        max_seq_len: int = 288,
        in_channels: int = 4,
        dim: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        learn_sigma: bool = True,
        qk_norm: bool = False,
        cap_feat_dim: int = 5120,
        rope_scaling_factor: float = 1.,
    ) -> None:
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        self.x_embedder = ColumnParallelLinear(
            in_features=patch_size * patch_size * in_channels,
            out_features=dim,
            bias=True,
            gather_output=True,
            init_method=nn.init.xavier_uniform_,
        )
        nn.init.constant_(self.x_embedder.bias, 0.)

        self.t_embedder = ParallelTimestepEmbedder(min(dim, 1024))
        self.cap_embedder = nn.Sequential(
            nn.LayerNorm(cap_feat_dim),
            ColumnParallelLinear(
                cap_feat_dim, min(dim, 1024), bias=True, gather_output=True,
                init_method=lambda x: nn.init.zeros_
                ),
        )

        self.layers = nn.ModuleList([
            TransformerBlock(layer_id, dim, n_heads, n_kv_heads, multiple_of,
                             ffn_dim_multiplier, norm_eps, qk_norm, cap_feat_dim)
            for layer_id in range(n_layers)
        ])
        self.final_layer = ParallelFinalLayer(dim, patch_size, self.out_channels)

        self.freqs_cis = DiT_Llama.precompute_freqs_cis(
            dim // n_heads, max_seq_len, rope_scaling_factor=rope_scaling_factor
        )
        self.eol_token = nn.Parameter(torch.empty(dim))
        self.pad_token = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.eol_token, std=0.02)
        nn.init.normal_(self.pad_token, std=0.02)

    def unpatchify(self, x: torch.Tensor, img_size: List[Tuple[int, int]], return_tensor=False) -> List[torch.Tensor]:
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        pH = pW = self.patch_size
        if return_tensor:
            H, W = img_size[0]
            B = x.size(0)
            L = (H // pH) * (W // pW + 1)  # one additional for eol
            x = x[:, :L].view(B, H // pH, W // pW + 1, pH, pW, self.out_channels)
            x = x[:, :, :-1]
            x = x.permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)
            return x
        else:
            imgs = []
            for i in range(x.size(0)):
                H, W = img_size[i]
                L = (H // pH) * (W // pW + 1)
                imgs.append(x[i][:L].view(
                    H // pH, W // pW + 1, pH, pW, self.out_channels
                )[:, :-1, :, :, :].permute(4, 0, 2, 1, 3).flatten(3, 4).flatten(1, 2))
        return imgs

    def patchify_and_embed(self, x: List[torch.Tensor] | torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        if isinstance(x, torch.Tensor):
            pH = pW = self.patch_size
            B, C, H, W = x.size()
            x = x.view(B, C, H // pH, pH, W // pW, pW).permute(0, 2, 4, 1, 3, 5).flatten(3)
            x = self.x_embedder(x)
            x = torch.cat([
                x,
                self.eol_token.view(1, 1, 1, -1).expand(B, H // pH, 1, -1),
            ], dim=2)
            x = x.flatten(1, 2)
            if x.size(1) < self.max_seq_len:
                x = torch.cat([
                    x,
                    self.pad_token.view(1, 1, -1).expand(B, self.max_seq_len - x.size(1), -1),
                ], dim=1)
            return x, [(H, W)] * B
        else:
            pH = pW = self.patch_size
            x_embed = []
            img_size = []

            for img in x:
                C, H, W = img.size()
                img_size.append((H, W))
                img = img.view(C, H // pH, pH, W // pW, pW).permute(1, 3, 0, 2, 4).flatten(2)
                img = self.x_embedder(img)
                img = torch.cat([
                    img,
                    self.eol_token.view(1, 1, -1).expand(H // pH, 1, -1),
                ], dim=1)
                img = img.flatten(0, 1)
                if img.size(0) < self.max_seq_len:
                    img = torch.cat([
                        img,
                        self.pad_token.view(1, -1).expand(self.max_seq_len - img.size(0), -1),
                    ], dim=0)
                x_embed.append(img)
            x_embed = torch.stack(x_embed, dim=0)
            return x_embed, img_size

    def forward(self, x, t, cap_feats, cap_mask):
        """
        Forward pass of DiT.
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x_is_tensor = isinstance(x, torch.Tensor)
        x, img_size = self.patchify_and_embed(x)
        self.freqs_cis = self.freqs_cis.to(x.device)

        t = self.t_embedder(t)                   # (N, D)
        cap_mask_float = cap_mask.float().unsqueeze(-1)
        cap_feats_pool = (cap_feats * cap_mask_float).sum(dim=1) / cap_mask_float.sum(dim=1)
        cap_emb = self.cap_embedder(cap_feats_pool)
        adaln_input = t + cap_emb

        for layer in self.layers:
            x = layer(
                x, cap_feats, cap_mask, self.freqs_cis[:x.size(1)],
                adaln_input=adaln_input
            )

        x = self.final_layer(x, adaln_input)
        x = self.unpatchify(x, img_size, return_tensor=x_is_tensor)         # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, cap_feats, cap_mask, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass
        for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, cap_feats, cap_mask)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, rope_scaling_factor: float = 1.0):
        """
        Precompute the frequency tensor for complex exponentials (cis) with
        given dimensions.

        This function calculates a frequency tensor with complex exponentials
        using the given dimension 'dim' and the end index 'end'. The 'theta'
        parameter scales the frequencies. The returned tensor contains complex
        values in complex64 data type.

        Args:
            dim (int): Dimension of the frequency tensor.
            end (int): End index for precomputing frequencies.
            theta (float, optional): Scaling factor for frequency computation.
                Defaults to 10000.0.

        Returns:
            torch.Tensor: Precomputed frequency tensor with complex
                exponentials.
        """
        freqs = 1.0 / (theta ** (
            torch.arange(0, dim, 2)[: (dim // 2)].float() / dim
        ))
        t = torch.arange(end, device=freqs.device, dtype=torch.float)  # type: ignore
        t = t / rope_scaling_factor
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    def parameter_count(self) -> int:
        tensor_parallel_module_list = (
            ColumnParallelLinear, RowParallelLinear, ParallelEmbedding,
        )
        total_params = 0

        def _recursive_count_params(module):
            nonlocal total_params
            is_tp_module = isinstance(module, tensor_parallel_module_list)
            for param in module.parameters(recurse=False):
                total_params += param.numel() * (
                    fs_init.get_model_parallel_world_size()
                    if is_tp_module else 1
                )
            for submodule in module.children():
                _recursive_count_params(submodule)

        _recursive_count_params(self)
        return total_params

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)


#############################################################################
#                                 DiT Configs                               #
#############################################################################


def DiT_Llama_600M_patch2(**kwargs):
    return DiT_Llama(
        patch_size=2, dim=1536, n_layers=16, n_heads=32, **kwargs
    )


def DiT_Llama_3B_patch2(**kwargs):
    return DiT_Llama(
        patch_size=2, dim=3072, n_layers=32, n_heads=32, **kwargs
    )


def DiT_Llama_7B_patch2(**kwargs):
    return DiT_Llama(
        patch_size=2, dim=4096, n_layers=32, n_heads=32, **kwargs
    )
