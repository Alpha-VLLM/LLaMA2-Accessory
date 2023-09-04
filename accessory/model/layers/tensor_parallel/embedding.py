from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_world_size,
)
from fairscale.nn.model_parallel.mappings import (
    copy_to_model_parallel_region,
    gather_from_model_parallel_region,
)
from .utils import init_tensor_parallel_weights


class ParallelEmbedding(nn.Module):
    r"""A tensor-parallel embedding layer. The output feature dimensions are
    divided among the tensor parallel ranks. Each part of the embeddings is
    calculated separately on each rank and gathered to form the complete
    embeddings.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        padding_idx (int, optional): If specified, the entries at
            :attr:`padding_idx` do not contribute to the gradient; therefore,
            the embedding vector at :attr:`padding_idx` is not updated during
            training, i.e. it remains as a fixed "pad". For a newly
            constructed Embedding, the embedding vector at :attr:`padding_idx`
            will default to all zeros, but can be updated to another value to
            be used as the padding vector.
        scale_grad_by_freq (bool, optional): If given, this will scale
            gradients by the inverse of frequency of the words in the
            mini-batch. Default ``False``.
        sparse (bool, optional): If ``True``, gradient w.r.t. :attr:`weight`
            matrix will be a sparse tensor. See Notes for more details
            regarding sparse gradients.
        init_fn (Callable[[torch.Tensor], Any], optional): Initializer function
            of the ``bias`` parameter. If set to ``None``, follows the default
            initialization of the PyTorch builtin ``torch.nn.Embedding`` layer.

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            (num_embeddings, embedding_dim) initialized from
            :math:`\mathcal{N}(0, 1)`

    Shape:
        - Input: :math:`(*)`, IntTensor or LongTensor of arbitrary shape
            containing the indices to extract
        - Output: :math:`(*, H)`, where `*` is the input shape and
            :math:`H=\text{embedding\_dim}`

    .. note::
        Keep in mind that only a limited number of optimizers support
        sparse gradients: currently it's :class:`optim.SGD` (`CUDA` and `CPU`),
        :class:`optim.SparseAdam` (`CUDA` and `CPU`) and :class:`optim.Adagrad`
        (`CPU`)

    .. note::
        The default initialization of the ``weight`` parameter is different in
        PyTorch and fairscale: The former uses ``torch.nn.init.normal_`` while
        the latter uses `torch.nn.init.xavier_normal_``. We follow the PyTorch
        default behavior.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        init_fn: Optional[Callable[[torch.Tensor], Any]] = None,
    ) -> None:
        super().__init__()
        self.num_emeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.init_fn = init_fn

        tp_world_size = get_model_parallel_world_size()
        assert self.embdding_dim % tp_world_size == 0, (
            "ParallelEmbedding currently requires that the embedding "
            "dimension is evenly divisible by the tensor parallel world size."
        )
        self.local_embeddding_dim = embedding_dim // tp_world_size

        self.weight = nn.Parameter(
            torch.empty([num_embeddings, self.local_embeddding_dim])
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_fn = self.init_fn or nn.init.normal_
        init_tensor_parallel_weights(self.weight, init_fn, 1)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        input_parallel = copy_to_model_parallel_region(input_)
        output_parallel = F.embedding(
            input_parallel,
            self.weight,
            self.padding_idx,
            None, 2.0,  # max_norm and norm_type, non-trivial to impl for tp.
            self.scale_grad_by_freq,
            self.sparse,
        )
        output = gather_from_model_parallel_region(output_parallel)
        return output

    def extra_repr(self) -> str:
        s = "{num_embeddings}, {embedding_dim}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)
