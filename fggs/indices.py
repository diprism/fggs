"""
# Algebraic index types

A tensor dimension often corresponds to a random variable, and indices along
the dimension often correspond to possible values of the variable.  Because the
variable can be of product or sum type, the range of indices [0,n) is often
[0,l*m) representing a Cartesian product, [0,l+m) representing a disjoint
union, or a nesting of these operations.  This algebraic type structure affects
our computations because our tensors are often (the result of einsum operations
involving) sparse factors that pack/unpack product/sum types, such as
eye(35).reshape(35,5,7) or eye(5,7).  We represent these sparse tensors
compactly for asymptotic speedups.

The key is the following language of "embeddings" (injective mappings) from
"physical" indices to "virtual" indices:

    Embedding ::= X(size)                      -- EmbeddingVar
                | Embedding * ... * Embedding  -- ProductEmbedding
                | size + Embedding + size      -- SumEmbedding

For example, to describe how a 5*7 "physical" matrix stores part of a
three-dimensional "virtual" tensor, we would use two EmbeddingVars [X(5), Y(7)]
to build up three embeddings such as [X(5) * Y(7), X(5), Y(7)].  The three
embeddings determine the shape of the virtual tensor to be [35, 5, 7].  Element
(i,j) of the matrix corresponds to element (i*7+j,i,j) of the tensor.  Other
elements of the tensor are presumed to be zero.

To take another example, to describe how a length-35 vector stores the second
diagonal of a 36*35 matrix, we would use one EmbeddingVar [Z(35)] to build up
two embeddings [1 + Z(35) + 0, Z(35)].  The two embeddings determine the shape
of the virtual tensor to be [36, 35].  Element k of the matrix corresponds to
element (1+k,k) of the matrix.  Other elements of the matrix are presumed to be
zero.

So, an EmbeddingVar X(5) represents a "physical" index in the range [0,5), and
an embedding expression represents an injective mapping from the physical
indices represented by its "free" EmbeddingVars to a virtual index.  The
embeddings define an affine transform from the storage offset and strides of
the virtual tensor (if it ever gets materialized) to the storage offset and
strides of the physical tensor (a view on the virtual).

In sum, a virtual tensor is represented by
- a physical tensor,
- a sequence of "physical" EmbeddingVars (ordered according to the dimensions
  of the physical tensor), and
- a sequence of "virtual" embeddings (ordered according to the dimensions of
  the virtual tensor) containing exactly those EmbeddingVars.
We store these three pieces of information together in an EmbeddedTensor.
"""

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

AntiSubst = Tuple[Dict[Tuple["Embedding", "Embedding"], "EmbeddingVar"],
                  Dict["EmbeddingVar", Tuple["Embedding", "Embedding"]]]

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
        warn(f"Attempt to unify {e} and {f} indicates index type mismatch")
        return False

    def antiunify(e: Embedding, f: Embedding, antisubst: AntiSubst) -> Embedding:
        """Antiunify the two embeddings by extending antisubst.  Return least
           general generalization (whose variables are all in antisubst)."""
        if e.size() != f.size():
            warn(f"Attempt to antiunify {e} and {f} indicates index type mismatch")
        if isinstance(e, ProductEmbedding) and isinstance(f, ProductEmbedding) and \
           len(e.factors) == len(f.factors):
            return ProductEmbedding(tuple(e1.antiunify(f1, antisubst)
                                          for (e1, f1) in zip(e.factors, f.factors)))
        if isinstance(e, SumEmbedding) and isinstance(f, SumEmbedding) and \
           e.before == f.before and e.after == f.after:
            return SumEmbedding(e.before, e.term.antiunify(f.term, antisubst), e.after)
        if (e, f) in antisubst[0]:
            return antisubst[0][(e, f)]
        new = EmbeddingVar(e.size())
        antisubst[0][(e, f)] = new
        antisubst[1][new] = (e, f)
        return new

@dataclass(eq=False, frozen=True) # identity matters
class EmbeddingVar(Embedding):
    _size: int

    def __repr__(self) -> str:
        return f"EmbeddingVar(size={self._size}, id={id(self)})"

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
        if len(self.pembeds) == len(self.vembeds) and \
           frozenset(self.pembeds) == \
           frozenset(forwarded_vembeds := tuple(e.forward(subst) for e in self.vembeds)):
            # vembeds is just a permutation of pembeds, so just clone view on physical
            return project(self.physical, forwarded_vembeds, self.pembeds, {})[0].clone()
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
    """
    To perform an einsum operation on EmbeddedTensors, we start with
    EmbeddedTensors whose sets of physical EmbeddingVars do not overlap, but
    then we *unify* the virtual embeddings that get co-indexed.  For example,
    if the two example EmbeddedTensors above were the inputs (bcd,ab->...),
    then we would unify X(5) * Y(7) with Z(35).  Hence, before we pass the
    einsum job to torch_semiring_einsum, we need to reshape (more generally,
    view) the second physical tensor -- a length-35 vector indexed by Z -- as a
    5*7 matrix indexed by X and Y.  If unification fails, then our einsum
    returns zero; the only possible unification failure in an einsum operation
    that respects the algebraic type structure of the indices should be between
    inl and inr (1 + Z(35) + 0 fails to unify with 0 + W(1) + 35).
    """
    if len(tensors) == 0:
        return semiring.from_int(1)
    pembeds_fv = set()
    index_to_vembed = {}
    subst = {}
    for (i, (tensor, input)) in enumerate(zip(tensors, inputs)):
        if pembeds_fv.isdisjoint(tensor.pembeds):
            pembeds_fv.update(tensor.pembeds)
        else:
            # TODO: freshen rather than error
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
    output_pembeds = tuple(frozenset(k for index in output
                                       for k in index_to_vembed[index].stride(subst)[1]))
    equation = ','.join(''.join(pembed_to_char[k] for k in pembeds)
                        for (view, pembeds) in projected_tensors) \
             + '->' + ''.join(pembed_to_char[k] for k in output_pembeds)
    compiled = torch_semiring_einsum.compile_equation(equation)
    out = semiring.einsum(compiled, *(view for (view, pembed) in projected_tensors))
    return EmbeddedTensor(out, output_pembeds, tuple(index_to_vembed[index] for index in output))

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

def add(t: EmbeddedTensor, u: EmbeddedTensor) -> EmbeddedTensor:
    """
    Add two EmbeddedTensors. We use anti-unification to compute how much they
    need to be expanded in order to match.
    """
    antisubst = ({}, {})
    lggs = tuple(e.antiunify(f, antisubst) for (e, f) in zip(t.vembeds, u.vembeds))
    (gs, es, fs) = zip(*((g, e, f) for (g, (e, f)) in antisubst[1].items()))
    t0 = EmbeddedTensor(t.physical, t.pembeds, es)
    u0 = EmbeddedTensor(u.physical, u.pembeds, fs)
    if t.physical.numel() >= u.physical.numel():
        td = t0.to_dense({})
        project(td, u.pembeds, fs, {})[0].add_(u.physical)
        return EmbeddedTensor(td, gs, lggs)
    else:
        ud = u0.to_dense({})
        project(ud, t.pembeds, es, {})[0].add_(t.physical)
        return EmbeddedTensor(ud, gs, lggs)

k1_   = EmbeddingVar(5)
phys2 = torch.randn(5,5)
virt2 = EmbeddedTensor(phys2, (k1,k1_), (SumEmbedding(0,ProductEmbedding(()),1),k1_,k1))
assert(torch.equal(add(virt, virt2).to_dense({}),
                   torch.add(virt.to_dense({}), virt2.to_dense({}))))
assert(torch.equal(add(virt2, virt).to_dense({}),
                   torch.add(virt2.to_dense({}), virt.to_dense({}))))

phys3 = torch.arange(0.0,500,10).reshape(5,2,5)
virt3 = EmbeddedTensor(phys3, (k1,k2,k1_), (k2,k1_,k1))
assert(torch.equal(add(virt, virt3).to_dense({}),
                   torch.add(virt.to_dense({}), virt3.to_dense({}))))
assert(torch.equal(add(virt3, virt).to_dense({}),
                   torch.add(virt3.to_dense({}), virt.to_dense({}))))
