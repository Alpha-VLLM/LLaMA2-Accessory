from typing import Optional, Tuple, Union, Dict, List
from importlib import resources as impresources
from dataclasses import dataclass, field
import math
import functools
import numpy as np

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
    copy_to_model_parallel_region,
    reduce_from_model_parallel_region
)

from ..components import RMSNorm
from transformers import Blip2Processor, Blip2Model, Blip2Config
import open_clip

import accessory
from accessory.configs import global_configs
if global_configs.USE_FLASH_ATTENTION:
    from flash_attn import flash_attn_func

default_linear_init = functools.partial(nn.init.kaiming_uniform_, a=math.sqrt(5))

from .llama import precompute_freqs_cis, apply_rotary_emb, repeat_kv

try:
    import megablocks.ops as ops
except ImportError:
    print("MegaBlocks not found, please see "
          "https://github.com/stanford-futuredata/megablocks/. "
          "Note that MegaBlocks depends on mosaicml-turbo, which only "
          "supports python 3.10.")
try:
    import stk
except ImportError:
    print(
        "STK not found: please see https://github.com/stanford-futuredata/stk")


@dataclass
class ModelArgs:
    dim: int = 4096
    hidden_dim: int = 16384
    head_dim: int = 128
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    norm_eps: float = 1e-5
    rope_theta: float = 1000000 # todo 1e6 really?

    max_batch_size: int = 32
    max_seq_len: int = 2048

    moe: Dict[str, int] = field(default_factory=lambda: {
        "num_experts_per_tok": 2,
        "num_experts": 8
    })
    load_balancing_weight: float = 0.01

    rope_scaling: Optional[float] = None

    load_pretrained_visual_encoder: bool = False


def promote_scalar(x: torch.Tensor) -> torch.Tensor:
    return x.view(1) if len(x.size()) == 0 else x


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

        self.flash = global_configs.USE_FLASH_ATTENTION
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


def _sparse_expert_merge(weights_to_merge: List[torch.Tensor], num_experts: int) -> torch.Tensor:
    weights_to_merge = [_.view(num_experts, -1, _.shape[-1]) for _ in weights_to_merge]
    weights_to_merge = torch.cat(weights_to_merge, dim=1)
    weights_to_merge = weights_to_merge.view(-1, weights_to_merge.shape[-1]).contiguous()
    return weights_to_merge

def _sparse_expert_split(weight_to_split: torch.Tensor, split_to: int, num_experts) -> List[torch.Tensor]:
    weight_to_split = weight_to_split.view(num_experts, -1, weight_to_split.shape[-1])
    l_split = list(torch.chunk(weight_to_split, split_to, dim=1))
    return l_split


class MoE(nn.Module):
    LOAD_BALANCING_LOSSES = []
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_tok: int,
    ):
        super().__init__()
        mp_size = fs_init.get_model_parallel_world_size()
        mp_rank = fs_init.get_model_parallel_rank()
        self.num_experts = num_experts

        self.dim = dim
        self.hidden_dim = hidden_dim
        assert hidden_dim % mp_size == 0
        self.hidden_dim_per_partition = hidden_dim // mp_size

        # experts
        # for every expert, each GPU holds its (1/mp_size)
        # todo init function
        self.w1 = nn.Parameter(
            torch.empty(self.hidden_dim_per_partition * self.num_experts,
                        self.dim))
        # set_weight_attrs(self.w1, {"weight_loader": self.moe_weight_loader})
        self.w2 = nn.Parameter(
            torch.empty(self.hidden_dim_per_partition * self.num_experts,
                        self.dim))
        # set_weight_attrs(self.w2, {"weight_loader": self.moe_weight_loader})
        self.w3 = nn.Parameter(
            torch.empty(self.hidden_dim_per_partition * self.num_experts,
                        self.dim))
        # set_weight_attrs(self.w3, {"weight_loader": self.moe_weight_loader})

        for w in [self.w1, self.w2, self.w3]:
            # mark as model parallel parameters,
            # otherwise the params will be broadcast within model parallel group to ensure consistency among ranks
            w.is_model_parallel = True
            # to support loading checkpoints saved with different model parallel size
            w.model_parallel_merge = functools.partial(_sparse_expert_merge, num_experts=self.num_experts)
            w.model_parallel_split = functools.partial(_sparse_expert_split, num_experts=self.num_experts)
            default_linear_init(w.data)

        self.gate = nn.Linear(dim, num_experts, bias=False)

        # Calculate the number of bits needed to represent the expert indices
        # so that we can pass it to radix sort.
        self.sort_end_bit = max(int(np.ceil(np.log2(self.num_experts))), 1)
        self.blocking = 128
        self.quantize_scatter_num_bits = -1

        # Calculate the number of bits needed to represent the column indices
        # in the intermediate sparse matrix.
        max_column_index = (self.hidden_dim * self.num_experts) // self.blocking
        self.transpose_sort_end_bit = max(
            int(np.ceil(np.log2(max_column_index))), 1)

        self.num_experts_per_tok = num_experts_per_tok

    def _load_balancing_loss(self, expert_scores, tokens_per_expert):
        """

        Args:
            expert_scores: size(n_tokens, num_experts), last dim sum to 1
            tokens_per_expert: (num_experts)

        Returns:

        """
        n_tokens = expert_scores.shape[0]
        assert not tokens_per_expert.requires_grad
        scores = expert_scores.mean(dim=0)
        scale = self.num_experts / (n_tokens * self.num_experts_per_tok)
        loss = scale * torch.dot(tokens_per_expert.to(scores), scores)
        return loss

    def sparse_transpose(
            self, size, row_indices,
            column_indices) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_columns = size[1] // self.blocking

        # Sort row indices by column indices to get the transposed matrix's
        # column indices.
        #
        # NOTE: Our sort operation uses the same width indices as the input
        # values. To avoid overflow when we have large activation matrices
        # we cast to 32-bit before sorting.
        _, gather_indices = ops.sort(column_indices.int(),
                                     self.transpose_sort_end_bit)

        # There are a constant number of blocks in every row of the sparse
        # matrix. A blocks offset is:
        #
        # row_index * blocks_per_row + column_index % blocks_per_row
        #
        # Once we have the block offsets ordered for transposition we can
        # divide by blocks_per_row to get the transposed column indices.
        column_indices_t = row_indices.gather(0, gather_indices.long())
        block_offsets_t = gather_indices.int()

        zero = torch.zeros((1, ), dtype=torch.int32, device=row_indices.device)
        nnz_per_column = ops.histogram(column_indices, block_columns)
        nnz_per_column = ops.inclusive_cumsum(nnz_per_column, 0)
        offsets_t = torch.cat([zero, nnz_per_column])
        return column_indices_t, offsets_t, block_offsets_t

    def topology(self, x: torch.Tensor,
                 padded_bins: torch.Tensor) -> stk.Matrix:
        padded_tokens, _ = x.size()
        assert padded_tokens % self.blocking == 0
        assert self.hidden_dim_per_partition % self.blocking == 0

        # Offsets for the sparse matrix. All rows have the
        # same number of nonzero blocks dictated by the
        # dimensionality of a single expert.
        block_rows = padded_tokens // self.blocking
        blocks_per_row = self.hidden_dim_per_partition // self.blocking
        offsets = torch.arange(
            0,
            block_rows * blocks_per_row + 1,
            blocks_per_row,
            dtype=torch.int32,
            device=x.device,
        )

        # Indices for the sparse matrix. The indices for
        # the intermediate matrix are dynamic depending
        # on the mapping of tokens to experts.
        column_indices = ops.topology(padded_bins, self.blocking, block_rows,
                                      blocks_per_row)

        # TODO(tgale): This is unused. Remove the need for this in stk.
        # For now, use meta init to save the device memory.
        data = torch.empty(
            column_indices.numel(),
            self.blocking,
            self.blocking,
            dtype=x.dtype,
            device="meta",
        )
        shape = (padded_tokens, self.hidden_dim_per_partition * self.num_experts)
        row_indices = stk.ops.row_indices(shape, data, offsets, column_indices)
        column_indices_t, offsets_t, block_offsets_t = self.sparse_transpose(
            shape, row_indices, column_indices)
        return stk.Matrix(
            shape,
            data,
            row_indices,
            column_indices,
            offsets,
            column_indices_t,
            offsets_t,
            block_offsets_t,
        )

    def indices_and_padded_bins(
        self, selected_experts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        selected_experts = selected_experts.int()
        bin_ids, indices = ops.sort(selected_experts, self.sort_end_bit)

        # Histogram the expert ids to identify the number of
        # tokens routed to each expert.
        tokens_per_expert = ops.histogram(selected_experts, self.num_experts)

        # Round the token counts up to the block size used in
        # the matrix muliplications. Caculate the starting
        # position of each bin.
        padded_tokens_per_expert = ops.round_up(tokens_per_expert,
                                                self.blocking)
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
        padded_bins = promote_scalar(padded_bins)

        # Calculate the bin bounds for the sorted tokens.
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = promote_scalar(bins)
        return indices, bin_ids, bins, padded_bins, tokens_per_expert


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (bsz, per_item_sequence_length, model_dim)
        sequence_length == bsz * per_item_sequence_length
        """
        input_shape = x.shape
        x = x.view(-1, input_shape[-1])

        x_for_score, x_for_ffn = x, copy_to_model_parallel_region(x)

        # <compute score>
        # gate_logits: (sequence_length, n_experts)
        gate_logits = self.gate(x_for_score)
        # all_probs: (sequence_length, n_experts) and upcast for softmax
        all_probs = F.softmax(gate_logits, dim=1, dtype=torch.float)
        # weights, selected_experts: (sequence_length, top-k)
        weights, selected_experts = torch.topk(all_probs, self.num_experts_per_tok, dim=-1)
        selected_experts = selected_experts.flatten()
        indices, bin_ids, bins, padded_bins, tokens_per_expert = self.indices_and_padded_bins(selected_experts)
        # todo maybe allreduce tokens_per_expert from data parallel group
        if self.training:
            MoE.LOAD_BALANCING_LOSSES.append(self._load_balancing_loss(all_probs, tokens_per_expert))
        weights /= weights.sum(dim=-1, keepdim=True)
        weights = weights.flatten().to(x_for_score.dtype)
        weights = copy_to_model_parallel_region(weights)


        # <compute ffn>
        # Permute tokens and pad to prepare expert computation
        # (top_k * sequence_length + padding, model_dim)
        x_for_ffn = ops.padded_gather(x_for_ffn, indices, bin_ids, bins, padded_bins,
                                      self.num_experts_per_tok)

        # Create the sparse matrix topology
        with torch.no_grad():
            topo = self.topology(x_for_ffn, padded_bins)

        # Perform the expert computation
        # First Dense x Dense -> Sparse for w1 and w3,
        # (top_k * sequence_length + padding, ffn_dim * n_experts)
        x_for_ffn = stk.Matrix(
            topo.size(),
            F.silu(stk.ops.sdd(x_for_ffn, self.w1.t(), topo).data) * stk.ops.sdd(x_for_ffn, self.w3.t(), topo).data,
            topo.row_indices,
            topo.column_indices,
            topo.offsets,
            topo.column_indices_t,
            topo.offsets_t,
            topo.block_offsets_t,
        )

        # Then Sparse x Dense -> Dense for w2
        # (top_k * sequence_length + padding, model_dim)
        x_for_ffn = stk.ops.dsd(x_for_ffn, self.w2)

        # todo why vllm code reduce at here? reduce after padded_scatter can cause lower communication
        # y = reduce_from_model_parallel_region(x_for_ffn.clone())
        # y = ops.padded_scatter(
        #     y,
        #     indices,
        #     bin_ids,
        #     weights,
        #     bins,
        #     padded_bins,
        #     self.num_experts_per_tok,
        #     self.quantize_scatter_num_bits,
        # )

        # <score meet ffn_output>
        # Permute back and remove padding
        # (top_k * sequence_length, model_dim)
        y = ops.padded_scatter(
            x_for_ffn,
            indices,
            bin_ids,
            weights,
            bins,
            padded_bins,
            self.num_experts_per_tok,
            self.quantize_scatter_num_bits,
        )

        y = reduce_from_model_parallel_region(y)

        return y.view(*input_shape)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = MoE(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            num_experts=args.moe['num_experts'],
            num_experts_per_tok=args.moe["num_experts_per_tok"],
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
    def __init__(self, args: ModelArgs, with_visual=False):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = ParallelEmbedding(
            args.vocab_size, args.dim, init_method=default_linear_init
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(layer_id, args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = ColumnParallelLinear(
            args.dim, args.vocab_size, bias=False, init_method=default_linear_init
        )

        self.freqs_cis = precompute_freqs_cis(
            self.args.dim // self.args.n_heads, self.args.max_seq_len * 2,
            theta=self.args.rope_theta, scaling=self.args.rope_scaling
        )

        self.image_words = 0
        self.cache_image_words = 0 # for inference
        if with_visual:

            default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float32)

            print("build llama model with qformerv2")
            if self.args.load_pretrained_visual_encoder:
                self.qformer = Blip2Model.from_pretrained(
                    "./blip2_opt2.7b", torch_dtype=self.norm.weight.dtype
                )
            else:
                self.qformer = Blip2Model(Blip2Config.from_pretrained(
                    str(impresources.files(accessory)/'resources/hf/Salesforce/blip2-opt-2.7b/config.json')))
            self.qformer.language_projection = None
            self.qformer.language_model = None
            self.qformer.to(self.norm.weight)

            print("build llama model with clip")
            if self.args.load_pretrained_visual_encoder:
                self.clip, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
            else:
                self.clip, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained=None)
            self.clip.transformer = None
            self.clip.to(self.norm.weight)

            print("build llama model with openclip")
            if self.args.load_pretrained_visual_encoder:
                self.openclip_convnext_xxl, _, _ = open_clip.create_model_and_transforms(
                    "convnext_xxlarge", pretrained="laion2b_s34b_b82k_augreg_soup"
                )
            else:
                self.openclip_convnext_xxl, _, _ = open_clip.create_model_and_transforms(
                    "convnext_xxlarge", pretrained=None
                )
            self.openclip_convnext_xxl = self.openclip_convnext_xxl.visual.trunk
            self.openclip_convnext_xxl.head.global_pool = nn.Identity()
            self.openclip_convnext_xxl.head.flatten = nn.Identity()
            self.openclip_convnext_xxl.to(self.norm.weight)

            print("build llama model with dinov2")
            if self.args.load_pretrained_visual_encoder:
                self.dinov2_vitg14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14", pretrained=True)
            else:
                self.dinov2_vitg14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14", pretrained=False)
            self.dinov2_vitg14.to(self.norm.weight)
            torch.set_default_dtype(default_dtype)

            self.qformer_proj = nn.Sequential(
                nn.Linear(768, args.dim),
                nn.LayerNorm(args.dim)
            )

            self.visual_proj = nn.Sequential(
                nn.Linear(3072 + 1024 + 1536, args.dim),
                nn.LayerNorm(args.dim),
            )

            self.image_words = (32 + 257 + 2) * 5
            self.image_size = 1024
            # add image tags
            self.start_img = nn.Parameter(torch.rand(1, 1, args.dim))
            self.end_img = nn.Parameter(torch.rand(1, 1, args.dim))


    def get_trainable_params(self):
        trainable = {}
        no_train_prefix = ["qformer.", "openclip_convnext_xxl.", "clip.", "dinov2_vitg14."]
        for name, para in self.named_parameters():
            if not any([name.startswith(_) for _ in no_train_prefix]):
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

        return x

    def encode_image(self, image):
        # images should be of size [bsz, 1024, 1024]
        self.qformer.eval()
        self.clip.eval()
        self.openclip_convnext_xxl.eval()
        self.dinov2_vitg14.eval()

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
            local_image_224 = F.interpolate(local_image.half(), size=(224,224), mode="bicubic").to(local_image)
            local_image_448 = F.interpolate(local_image.half(), size=(448,448), mode="bicubic").to(local_image)
            local_parts_224 = [
                local_image_448[..., :224, :224], local_image_448[..., :224, 224:],
                local_image_448[..., 224:, :224], local_image_448[..., 224:, 224:]
            ]
            local_224 = torch.stack([local_image_224] + local_parts_224, dim=1)
            local_224 = local_224.view(-1, *local_224.shape[2:])

            local_image_512 = F.interpolate(local_image.half(), size=(512,512), mode="bicubic").to(local_image)
            local_parts_512 = [
                local_image[..., :512, :512], local_image[..., :512, 512:],
                local_image[..., 512:, :512], local_image[..., 512:, 512:]
            ]
            local_512 = torch.stack([local_image_512] + local_parts_512, dim=1)
            local_512 = local_512.view(-1, *local_512.shape[2:])

            local_image_feats = self.qformer.get_qformer_features(pixel_values=local_224).last_hidden_state
            image_feats = torch.zeros([padded_image_bs*5, *local_image_feats.size()[1:]],
                                      device=local_image_feats.device, dtype=local_image_feats.dtype)
            dist.all_gather_into_tensor(image_feats, local_image_feats, group=fs_init.get_model_parallel_group())

            local_clip_image_feats = self.clip_encode_image(local_224)
            local_convnext_image_feats = self.openclip_convnext_xxl(local_512)
            assert local_convnext_image_feats.size()[1:] == (3072, 16, 16)
            local_convnext_image_feats = local_convnext_image_feats.flatten(-2).permute(0, 2, 1)  # (*, 256, 3072)
            local_convnext_image_feats = torch.cat([
                local_convnext_image_feats.mean(dim=1, keepdim=True),  # add gap as cls token
                local_convnext_image_feats,
            ], dim=1)  # (*, 257, 3072)

            clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073])
            clip_mean = clip_mean.to(local_image, non_blocking=True).view(3, 1, 1)
            clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711])
            clip_std = clip_std.to(local_image, non_blocking=True).view(3, 1, 1)
            dinov2_mean = torch.Tensor([0.485, 0.456, 0.406]).to(local_image, non_blocking=True).view(3, 1, 1)
            dinov2_std = torch.Tensor([0.229, 0.224, 0.225]).to(local_image, non_blocking=True).view(3, 1, 1)
            local_dinov2_image_feats = self.dinov2_vitg14.forward_features(
                (local_224 * clip_std + clip_mean - dinov2_mean) / dinov2_std
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

            ens_image_feats = torch.zeros([padded_image_bs*5, *local_ens_image_feats.size()[1:]],
                                          device=local_ens_image_feats.device, dtype=local_ens_image_feats.dtype)
            dist.all_gather_into_tensor(ens_image_feats, local_ens_image_feats,
                                        group=fs_init.get_model_parallel_group())

            ens_image_feats = ens_image_feats[:image_bs*5]
            image_feats = image_feats[:image_bs*5]

        image_feats = self.qformer_proj(image_feats)
        ens_image_feats = self.visual_proj(ens_image_feats)
        image_feats = torch.cat([image_feats, ens_image_feats], dim=1)
        # image_feats = torch.zeros([image.size(0), 32, 768], dtype=torch.half, device=image.device)
        # image_feats = self.qformer_proj(image_feats)

        image_feats = image_feats.view(image_bs, 5, *image_feats.shape[1:])
        image_feats = list(torch.unbind(image_feats, dim=1))
        return image_feats

    def forward(self, examples, image=None):
        self._destroy_kv_cache()  # training always disables kv cache
        MoE.LOAD_BALANCING_LOSSES.clear()

        _bsz, seqlen = examples.shape
        h = self.tok_embeddings(examples)
        self.freqs_cis = self.freqs_cis.to(h.device)

        image_words = 0
        if image is not None:
            h_bos, h_caption = h[:, :1], h[:, 1:]
            l_image_tokens: List = self.encode_image(image)
            for i, image_tokens in enumerate(l_image_tokens):
                image_tokens = torch.cat((self.start_img.expand(_bsz, -1, -1),
                                          image_tokens,
                                          self.end_img.expand(_bsz, -1, -1)), dim=1)
                l_image_tokens[i] = image_tokens
            image_tokens = torch.cat(l_image_tokens, dim=1)
            image_words = image_tokens.shape[1]
            assert image_words == self.image_words, f"{image_words} v.s. {self.image_words}, {[_.shape for _ in l_image_tokens]}"
            h = torch.cat((h_bos, image_tokens, h_caption), dim=1)
            seqlen = h.shape[1]

        freqs_cis = self.freqs_cis[:seqlen]
        for layer in self.layers:
            h = layer(h, start_pos=0, freqs_cis=freqs_cis, mask="causal")
        h = self.norm(h)
        output = self.output(h[:, image_words:, :])

        additional_loss_dict = {}
        if self.training:
            load_balancing_loss = sum(MoE.LOAD_BALANCING_LOSSES) / max(len(MoE.LOAD_BALANCING_LOSSES), 1)
            additional_loss_dict['load_balancing'] = (load_balancing_loss, self.args.load_balancing_weight)
        return output, additional_loss_dict


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
            l_image_tokens: List = self.encode_image(image)
            for i, image_tokens in enumerate(l_image_tokens):
                image_tokens = torch.cat((self.start_img.expand(_bsz, -1, -1),
                                          image_tokens,
                                          self.end_img.expand(_bsz, -1, -1)), dim=1)
                l_image_tokens[i] = image_tokens
            image_tokens = torch.cat(l_image_tokens, dim=1)
            self.cache_image_words = image_tokens.shape[1]
            assert self.cache_image_words == self.image_words
            h = torch.cat((h_bos, image_tokens, h_caption), dim=1).to(h_bos)
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
            layer.attention.allocate_kv_cache(max_batch_size, self.args.max_seq_len)

    def _destroy_kv_cache(self) -> None:
        for layer in self.layers:
            layer.attention.destroy_kv_cache()

    def get_quant_blocklist(self) -> List[str]:
        vision_prefixes = [
            "clip.", "openclip_convnext_xxl.", "dinov2_vitg14.", "qformer.",
            "visual_proj.", "qformer_proj.",
        ]
        blocklist = []
        for n, m in self.named_modules():
            if any(n.startswith(x) for x in vision_prefixes):
                blocklist.append(n)
        return blocklist
