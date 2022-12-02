__all__ = ['DiagonalTensor']
from typing import Tuple, Sequence
import torch
from torch import Tensor

def embed(tensor: Tensor, mapping: Tensor):
    ret = tensor.new_zeros(torch.tensor(tensor.size())
                                .gather(0, mapping)
                                .tolist())
    stride = torch.tensor(ret.stride())
    ret.as_strided(tensor.size(),
                   stride.new_zeros(tensor.ndim)
                         .scatter_add_(0, mapping, stride)
                         .tolist())[:] = tensor
    return ret

class DiagonalTensor:
    """A wrapper around Tensor that maps virtual dimensions to physical ones surjectively."""
    def __init__(self, tensor: Tensor, mapping: Sequence[int]):
        if frozenset(range(0, tensor.ndim)) != frozenset(d for d in mapping):
            raise ValueError(f"The mapping {mapping} is not surjective onto {tensor.size()}")
        self.tensor = tensor
        self.mapping = torch.tensor(mapping)

    def to_dense(self) -> Tensor:
        return embed(self.tensor, self.mapping)

    def to_denser(self, mapping: Sequence[int]):
        if len(self.mapping) != len(mapping):
            raise ValueError(f"The number of virtual dimensions differ between the old mapping {self.mapping} and the new mapping {mapping}")
        new_mapping = torch.tensor(mapping)
        new_ndim = new_mapping.max().item() + 1
        if frozenset(range(0, new_ndim)) != frozenset(d for d in mapping):
            raise ValueError(f"The new mapping {mapping} is not surjective onto {new_ndim} physical dimensions")
        conversion = self.mapping.new_full((new_ndim,), -1) \
                                 .scatter_reduce_(0, new_mapping, self.mapping, "amax")
        if not torch.equal(conversion, self.mapping.new_full((new_ndim,), self.tensor.ndim)
                                           .scatter_reduce_(0, new_mapping, self.mapping, "amin")):
            raise ValueError(f"The new mapping {mapping} is not a refinement of the old mapping {self.mapping}")
        return DiagonalTensor(embed(self.tensor, conversion), mapping)

dt1 = DiagonalTensor(torch.randn(5,2), (1,0,0))
dt2 = dt1.to_denser((1,2,0))
assert(torch.equal(dt1.to_dense(), dt2.to_dense()))
