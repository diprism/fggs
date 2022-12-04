# Algebraic index types

from __future__ import annotations
from typing import Sequence, Tuple, Dict, Set, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from warnings import warn
from math import prod
import torch
from torch import Tensor
import torch_semiring_einsum
from semirings import RealSemiring

Subst = Dict["EmbeddingVar", "Embedding"]

class Embedding(ABC):
    """An injective mapping from physical indices to virtual indices."""

    @abstractmethod
    def size(self) -> int:
        """The virtual indices in the image of this mapping lie in [0,size())."""
        pass

    @abstractmethod
    def stride(subst: Subst) -> Tuple[int, Dict[EmbeddingVar, int]]:
        """The coefficients of the affine map from physical to virtual indices."""
        pass

    def forward(self, subst: Subst) -> Embedding:
        """Look in subst for the end of the forwarding chain starting with self."""
        return self

    def unify(e: Embedding, f: Embedding, subst: Subst) -> bool:
        """Unify the two embeddings by extending subst. Return success."""
        e = e.forward(subst)
        f = f.forward(subst)
        if e is f: return True
        if isinstance(e, ProductEmbedding) and isinstance(f, ProductEmbedding):
            if len(e.factors) == len(f.factors):
                for (e1, f1) in zip(e.factors, f.factors):
                    if not e1.unify(f1, subst): return False
                return True
            else:
                warn(f"Attempt to unify {e} and {f} indicates index type mismatch")
                return False
        if isinstance(e, SumEmbedding) and isinstance(f, SumEmbedding):
            if e.before == f.before and e.after == f.after:
                return e.term.unify(f.term, subst)
            else:
                ets = e.term.size()
                fts = f.term.size()
                if e.before + ets + e.after != f.before + fts + f.after \
                   or e.before + ets > f.before and f.before + fts > e.before:
                    warn(f"Attempt to unify {e} and {f} indicates index type mismatch")
                return False
        if e.size() != f.size():
            warn(f"Attempt to unify {e} and {f} indicates index type mismatch")
        if isinstance(e, EmbeddingVar):
            subst[e] = f
            return True
        if isinstance(f, EmbeddingVar):
            subst[f] = e
            return True
        return False

@dataclass(eq=False, frozen=True) # identity matters
class EmbeddingVar(Embedding):
    _size: int

    def size(self):
        return self._size

    def stride(self, subst):
        fwd = self.forward(subst)
        return (0, {self: 1}) if fwd is self else fwd.stride(subst)

    def forward(self, subst):
        if self in subst:
            subst[self] = ret = subst[self].forward(subst)
            return ret
        else:
            return self

@dataclass(frozen=True)
class ProductEmbedding(Embedding):
    factors: Sequence[Embedding]

    def size(self):
        return prod(e.size() for e in self.factors)

    def stride(self, subst):
        offset = 0
        stride = {}
        for e in self.factors:
            if offset or stride:
                n = e.size()
                offset *= n
                for k in stride: stride[k] *= n
            (o, s) = e.stride(subst)
            offset += o
            for k in s: stride[k] = stride.get(k, 0) + s[k]
        return (offset, stride)

@dataclass(frozen=True)
class SumEmbedding(Embedding):
    before: int
    term: Embedding
    after: int

    def size(self):
        return self.before + self.term.size() + self.after

    def stride(self, subst):
        (o, s) = self.term.stride(subst)
        return (o + self.before, s)

def project(virtual: Tensor,
            pembeds: Optional[Sequence[EmbeddingVar]],
            vembeds: Sequence[Embedding],
            subst: Subst) -> Tuple[Tensor, Sequence[EmbeddingVar]]:
    """Extract a view of the given tensor, so that indexing into the returned
       tensor according to pembeds is equivalent to indexing into the given
       tensor according to vembeds."""
    if virtual.size() != tuple(e.size() for e in vembeds):
        raise ValueError(f"project(tensor of size {virtual.size()}, ..., vembeds of size {tuple(e.size() for e in vembeds)}")
    offset = virtual.storage_offset()
    stride = {}
    for (e, n) in zip(vembeds, virtual.stride()):
        (o, s) = e.stride(subst)
        offset += o * n
        for k in s: stride[k] = stride.get(k, 0) + s[k] * n
    if pembeds is None:
        pembeds = tuple(stride.keys())
    else:
        vembeds_fv = frozenset(stride.keys())
        pembeds_fv = frozenset(pembeds)
        if vembeds_fv != pembeds_fv:
            raise ValueError(f"project(..., pembeds with {pembeds_fv}, vembeds with {vembeds_fv})")
    return (virtual.as_strided(tuple(k.size() for k in pembeds),
                               tuple(stride[k] for k in pembeds),
                               offset),
            pembeds)

@dataclass(frozen=True)
class EmbeddedTensor:
    physical: Tensor
    pembeds: Sequence[EmbeddingVar]
    vembeds: Sequence[Embedding]

    def __post_init__(self):
        if self.physical.size() != tuple(k.size() for k in self.pembeds):
            raise ValueError(f"EmbeddedTensor(tensor of size {self.physical.size()}, pembeds of size {tuple(k.size() for k in self.pembeds)}, ...)")

    def to_dense(self, subst: Subst) -> Tensor:
        """Expand a physical tensor to a mostly-zero virtual tensor."""
        virtual = self.physical.new_zeros(tuple(e.size() for e in self.vembeds))
        # TODO: allow pembeds_fv <= vembeds_fv by repeating self.physical?
        project(virtual, self.pembeds, self.vembeds, subst)[0].copy_(self.physical)
        return virtual

k1 = EmbeddingVar(5)
k2 = EmbeddingVar(2)
phys = torch.randn(5,2)
virt = EmbeddedTensor(phys, (k1,k2), (k2,k1,k1))
assert(torch.equal(virt.to_dense({}),
                   phys.t().diag_embed(dim1=1, dim2=2)))

ones = torch.ones(2,3)
k3 = EmbeddingVar(3)
diag = EmbeddedTensor(ones, (k2,k3), (k2,k3,SumEmbedding(1,ProductEmbedding((k2,k3)),2)))
assert(torch.equal(diag.to_dense({}),
                   torch.tensor([[[0,1,0,0,0,0,0,0,0],
                                  [0,0,1,0,0,0,0,0,0],
                                  [0,0,0,1,0,0,0,0,0]],
                                 [[0,0,0,0,1,0,0,0,0],
                                  [0,0,0,0,0,1,0,0,0],
                                  [0,0,0,0,0,0,1,0,0]]])))

k4 = EmbeddingVar(6)
subst = {}
assert(SumEmbedding(1,k4,2).unify(SumEmbedding(1,ProductEmbedding((k2,k3)),2), subst) and
       subst == {k4: ProductEmbedding((k2,k3))} and
       SumEmbedding(1,k4,2).unify(SumEmbedding(1,ProductEmbedding((k2,k3)),2), subst) and
       subst == {k4: ProductEmbedding((k2,k3))})
subst = {}
assert(SumEmbedding(1,ProductEmbedding((k2,k3)),2).unify(SumEmbedding(1,k4,2), subst) and
       subst == {k4: ProductEmbedding((k2,k3))} and
       SumEmbedding(1,ProductEmbedding((k2,k3)),2).unify(SumEmbedding(1,k4,2), subst) and
       subst == {k4: ProductEmbedding((k2,k3))})
subst = {}
assert(not SumEmbedding(1,k4,2).unify(SumEmbedding(7,k2,0), subst))
k2_ = EmbeddingVar(2)
k3_ = EmbeddingVar(3)
assert(SumEmbedding(1,ProductEmbedding((k2_,k3_)),2).unify(SumEmbedding(1,ProductEmbedding((k2,k3)),2), subst) and
       subst == {k2_: k2, k3_: k3})

def einsum(tensors: Sequence[EmbeddedTensor],
           inputs: Sequence[Sequence[Any]], 
           output: Sequence[Any],
           semiring: Semiring) -> EmbeddedTensor:
    if len(tensors) == 0:
        return semiring.from_int(1)
    pembeds_fv = set()
    index_to_vembed = {}
    subst = {}
    for (i, (tensor, input)) in enumerate(zip(tensors, inputs)):
        if pembeds_fv.isdisjoint(tensor.pembeds):
            pembeds_fv.update(tensor.pembeds)
        else:
            raise ValueError(f"einsum(tensor {i} whose pembeds are not disjoint, ..., ...)")
        if len(tensor.vembeds) != len(input):
            raise ValueError(f"einsum(tensor {i} with {len(tensor.vembeds)} virtual dimensions, input {i} with {len(input)} indices, ...)")
        for (vembed, index) in zip(tensor.vembeds, input):
            if index in index_to_vembed:
                if not index_to_vembed[index].unify(vembed, subst):
                    return semiring.from_int(0)
            else:
                index_to_vembed[index] = vembed
    projected_tensors = [project(tensor.physical, None, tensor.pembeds, subst)
                         for tensor in tensors]
    pembed_to_char = {k: chr(ord('a') + i)
                      for i, k in enumerate(frozenset(k for (view, pembeds) in projected_tensors
                                                        for k in pembeds))}
    output_pembeds = list(frozenset(k for index in output
                                      for k in index_to_vembed[index].stride(subst)[1]))
    equation = ','.join(''.join(pembed_to_char[k] for k in pembeds)
                        for (view, pembeds) in projected_tensors) \
             + '->' + ''.join(pembed_to_char[k] for k in output_pembeds)
    compiled = torch_semiring_einsum.compile_equation(equation)
    out = semiring.einsum(compiled, *(view for (view, pembed) in projected_tensors))
    return EmbeddedTensor(out, output_pembeds, list(index_to_vembed[index] for index in output))

matrix = torch.randn(36)
vector = torch.randn(7)
k5 = EmbeddingVar(5)
k5_= EmbeddingVar(5)
k6 = EmbeddingVar(35)
k7 = EmbeddingVar(7)
k7_= EmbeddingVar(7)
k8 = EmbeddingVar(36)
semiring = RealSemiring(dtype=matrix.dtype, device=matrix.device)
assert(1e-5 > (matrix[1:].reshape((5,7)).matmul(vector) -
               einsum([EmbeddedTensor(matrix, (k8,), (k8,)),
                       EmbeddedTensor(vector, (k7,), (k7,)),
                       # Here's a sum-type factor represented compactly:
                       EmbeddedTensor(torch.tensor(1).unsqueeze_(0).expand([35]),
                                      (k6,),
                                      (SumEmbedding(1,k6,0),k6)),
                       # Here's a product-type factor represented compactly:
                       EmbeddedTensor(torch.tensor(1).unsqueeze_(0).unsqueeze_(0).expand([5,7]),
                                      (k5_,k7_),
                                      (ProductEmbedding((k5_,k7_)),k5_,k7_))],
                      [["maybe-f"], ["i"], ["maybe-f","f"], ["f","o","i"]],
                      ["o"],
                      semiring).to_dense({})).abs().max())
