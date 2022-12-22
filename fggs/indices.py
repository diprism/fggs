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

    Embedding ::= X(numel)                     -- EmbeddingVar
                | Embedding * ... * Embedding  -- ProductEmbedding
                | numel + Embedding + numel    -- SumEmbedding

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
from typing import Sequence, Tuple, Dict, Set, Any, Optional, Union, cast
from dataclasses import dataclass
from abc import ABC, abstractmethod
from warnings import warn
from functools import reduce
from operator import mul
import torch
from torch import Tensor, Size
import torch_semiring_einsum
from fggs.semirings import Semiring, RealSemiring

Rename = Dict["EmbeddingVar", "EmbeddingVar"]

Subst = Dict["EmbeddingVar", "Embedding"]

AntiSubst = Tuple[Dict[Tuple["Embedding", "Embedding"], "EmbeddingVar"],
                  Dict["EmbeddingVar", Tuple["Embedding", "Embedding"]]]

class Embedding(ABC):
    """An injective mapping from physical indices to virtual indices."""

    @abstractmethod
    def numel(self) -> int:
        """The virtual indices in the image of this mapping lie in [0,numel())."""
        pass

    @abstractmethod
    def stride(self, subst: Subst) -> Tuple[int, Dict[EmbeddingVar, int]]:
        """The coefficients of the affine map from physical to virtual indices."""
        pass

    def forward(self, subst: Subst) -> Embedding:
        """Look in subst for the end of the forwarding chain starting with self."""
        return self

    @abstractmethod
    def alpha(e: Embedding, f: Embedding, rename: Rename) -> bool:
        """Check whether the given renaming turns e into f.  Each EmbeddingVar
           in e must be a key in the given renaming, even if it's also in f."""
        pass

    @abstractmethod
    def freshen(self, rename: Rename) -> Embedding:
        """Copy self while performing the given renaming, extending the
           renaming whenever necessary."""
        pass

    def unify(e: Embedding, f: Embedding, subst: Subst) -> bool:
        """Unify the two embeddings by extending subst. Return success."""
        e = e.forward(subst)
        f = f.forward(subst)
        if e is f: return True
        if isinstance(e, ProductEmbedding) and isinstance(f, ProductEmbedding):
            if len(e.factors) == len(f.factors):
                return all(e1.unify(f1, subst) for e1, f1 in zip(e.factors, f.factors))
            else:
                warn(f"Attempt to unify {e} and {f} indicates index type mismatch")
                return False
        if isinstance(e, SumEmbedding) and isinstance(f, SumEmbedding):
            if e.before == f.before and e.after == f.after:
                return e.term.unify(f.term, subst)
            else:
                etn = e.term.numel()
                ftn = f.term.numel()
                if e.before + etn + e.after != f.before + ftn + f.after \
                   or e.before + etn > f.before and f.before + ftn > e.before:
                    warn(f"Attempt to unify {e} and {f} indicates index type mismatch")
                return False
        if e.numel() != f.numel():
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
        if e.numel() != f.numel():
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
        new = EmbeddingVar(e.numel())
        antisubst[0][(e, f)] = new
        antisubst[1][new] = (e, f)
        return new

@dataclass(eq=False, frozen=True) # identity matters
class EmbeddingVar(Embedding):
    _numel: int

    def __repr__(self) -> str:
        return f"EmbeddingVar(numel={self._numel}, id={id(self)})"

    def numel(self):
        return self._numel

    def stride(self, subst):
        fwd = self.forward(subst)
        return (0, {self: 1}) if fwd is self else fwd.stride(subst)

    def forward(self, subst):
        if self in subst:
            subst[self] = ret = subst[self].forward(subst)
            return ret
        else:
            return self

    def alpha(e, f, rename):
        return rename.get(e, None) is f

    def freshen(self, rename):
        if self in rename:
            return rename[self]
        else:
            rename[self] = ret = EmbeddingVar(self._numel)
            return ret

@dataclass(frozen=True)
class ProductEmbedding(Embedding):
    factors: Sequence[Embedding]

    def numel(self):
        return reduce(mul, (e.numel() for e in self.factors), 1)

    def stride(self, subst):
        offset = 0
        stride = {}
        for e in self.factors:
            if offset or stride:
                n = e.numel()
                offset *= n
                for k in stride: stride[k] *= n
            (o, s) = e.stride(subst)
            offset += o
            for k in s: stride[k] = stride.get(k, 0) + s[k]
        return (offset, stride)

    def alpha(e, f, rename):
        return isinstance(f, ProductEmbedding) and \
               len(e.factors) == len(f.factors) and \
               all(e1.alpha(f1, rename) for e1, f1 in zip(e.factors, f.factors))

    def freshen(self, rename):
        return ProductEmbedding(tuple(e.freshen(rename) for e in self.factors))

@dataclass(frozen=True)
class SumEmbedding(Embedding):
    before: int
    term: Embedding
    after: int

    def numel(self):
        return self.before + self.term.numel() + self.after

    def stride(self, subst):
        (o, s) = self.term.stride(subst)
        return (o + self.before, s)

    def alpha(e, f, rename):
        return isinstance(f, SumEmbedding) and \
               e.before == f.before and e.after == f.after and \
               e.term.alpha(f.term, rename)

    def freshen(self, rename):
        return SumEmbedding(self.before, self.term.freshen(rename), self.after)

def project(virtual: Tensor,
            pembeds: Optional[Sequence[EmbeddingVar]],
            vembeds: Sequence[Embedding],
            subst: Subst) -> Tuple[Tensor, Sequence[EmbeddingVar]]:
    """Extract a view of the given tensor, so that indexing into the returned
       tensor according to pembeds is equivalent to indexing into the given
       tensor according to vembeds."""
    if virtual.size() != Size(e.numel() for e in vembeds):
        raise ValueError(f"project(tensor of {virtual.size()}, ..., vembeds of {Size(e.numel() for e in vembeds)}")
    offset = virtual.storage_offset()
    stride : Dict[EmbeddingVar, int] = {}
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
    return (virtual.as_strided(Size(k.numel() for k in pembeds),
                               tuple(stride[k] for k in pembeds),
                               offset),
            pembeds)

@dataclass
class EmbeddedTensor:
    physical: Tensor
    pembeds: Sequence[EmbeddingVar] = cast(Sequence[EmbeddingVar], None) #-/
    vembeds: Sequence[Embedding]    = cast(Sequence[Embedding],    None) #-/

    def __post_init__(self):
        # Default to the trivial (dense) embedding, to convert from Tensor
        if self.pembeds is None:
            self.pembeds = tuple(EmbeddingVar(n) for n in self.physical.size())
        elif self.physical.size() != Size(k.numel() for k in self.pembeds):
            raise ValueError(f"EmbeddedTensor(tensor of {self.physical.size()}, pembeds of {Size(k.numel() for k in self.pembeds)}, ...)")
        if self.vembeds is None:
            self.vembeds = self.pembeds

    @property
    def shape(self):
        return self.size()

    def size(self) -> Size:
        return Size(e.numel() for e in self.vembeds)

    def freshen(self) -> EmbeddedTensor:
        """Return a new EmbeddedTensor (with same underlying physical storage)
           with fresh EmbeddingVars."""
        rename : Rename = {}
        return EmbeddedTensor(self.physical,
                              tuple(k.freshen(rename) for k in self.pembeds),
                              tuple(e.freshen(rename) for e in self.vembeds))

    def clone(self) -> EmbeddedTensor:
        rename : Rename = {}
        return EmbeddedTensor(self.physical.clone(),
                              tuple(k.freshen(rename) for k in self.pembeds),
                              tuple(e.freshen(rename) for e in self.vembeds))

    def copy_(self, src: EmbeddedTensor) -> None:
        """Make self equal to src, possibly by overwriting self."""
        p = self.physical
        if p.numel() == src.physical.numel() and \
           p.dtype   == src.physical.dtype: # Can we reuse self.physical?
            l = list(enumerate(p.stride()))
            l.sort(reverse = True, key = lambda pair: pair[1])
            p = p.permute(tuple(dim for dim, _ in l))
            if p.is_contiguous():
                self.physical = p.view(src.physical.size()).copy_(src.physical)
            else:
                self.physical = src.physical.clone()
        else:
            self.physical = src.physical.clone()
        rename : Rename = {}
        self.pembeds = tuple(k.freshen(rename) for k in src.pembeds)
        self.vembeds = tuple(e.freshen(rename) for e in src.vembeds)

    def isdisjoint(self, other: Union[Set[EmbeddingVar], EmbeddedTensor]) -> bool:
        """Check whether the EmbeddingVars in self and other overlap."""
        return (frozenset(other.pembeds) if isinstance(other, EmbeddedTensor) else other) \
               .isdisjoint(self.pembeds)

    def to_dense(self, subst: Subst) -> Tensor:
        """Expand a physical tensor to a mostly-zero virtual tensor."""
        if len(self.pembeds) == len(self.vembeds):
            forwarded_vembeds = tuple(e.forward(subst) for e in self.vembeds)
            if frozenset(self.pembeds) == frozenset(forwarded_vembeds):
                # vembeds is just a permutation of pembeds, so just clone view on physical
                return project(self.physical,
                               cast(Tuple[EmbeddingVar], forwarded_vembeds),
                               self.pembeds, {})[0].clone()
        virtual = self.physical.new_zeros(self.size())
        # TODO: allow pembeds_fv <= vembeds_fv by repeating self.physical?
        project(virtual, self.pembeds, self.vembeds, subst)[0].copy_(self.physical)
        return virtual

    def equal(self, other: EmbeddedTensor) -> bool:
        if self.size() != other.size(): return False
        if not self.isdisjoint(other): other = other.freshen()
        selfzero  = self .physical == 0
        otherzero = other.physical == 0
        # Compare overlapping parts to each other
        subst : Subst = {}
        if all(v.unify(u, subst) for v, u in zip(self.vembeds, other.vembeds)):
            # There is overlap
            subself, subembeds = project(self.physical, None, self.pembeds, subst)
            subother, _ = project(other.physical, subembeds, other.pembeds, subst)
            if not subself.equal(subother): return False
            project(selfzero , None, self .pembeds, subst)[0].fill_(True)
            project(otherzero, None, other.pembeds, subst)[0].fill_(True)
        # Compare non-overlapping parts to zero
        return bool(selfzero.all()) and bool(otherzero.all())

    def allclose(self, other: EmbeddedTensor, rtol=1e-05, atol=1e-08, equal_nan=False) -> bool:
        if self.size() != other.size(): return False
        if not self.isdisjoint(other): other = other.freshen()
        selfzero  = (self .physical >= -atol).min(self .physical <= atol)
        otherzero = (other.physical >= -atol).min(other.physical <= atol)
        # Compare overlapping parts to each other
        subst : Subst = {}
        if all(v.unify(u, subst) for v, u in zip(self.vembeds, other.vembeds)):
            # There is overlap
            subself, subembeds = project(self.physical, None, self.pembeds, subst)
            subother, _ = project(other.physical, subembeds, other.pembeds, subst)
            if not subself.allclose(subother, rtol=rtol, atol=atol, equal_nan=equal_nan): return False
            project(selfzero , None, self .pembeds, subst)[0].fill_(True)
            project(otherzero, None, other.pembeds, subst)[0].fill_(True)
        # Compare non-overlapping parts to zero
        return bool(selfzero.all()) and bool(otherzero.all())

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
    pembeds_fv : Set[EmbeddingVar] = set()
    index_to_vembed : Dict[Any, Embedding] = {}
    subst : Subst = {}
    freshened_tensors = []
    for (i, (tensor, input)) in enumerate(zip(tensors, inputs)):
        if not tensor.isdisjoint(pembeds_fv): tensor = tensor.freshen()
        freshened_tensors.append(tensor)
        pembeds_fv.update(tensor.pembeds)
        if len(tensor.vembeds) != len(input):
            raise ValueError(f"einsum(tensor {i} with {len(tensor.vembeds)} virtual dimensions, input {i} with {len(input)} indices, ...)")
        for (vembed, index) in zip(tensor.vembeds, input):
            if index in index_to_vembed:
                if not index_to_vembed[index].unify(vembed, subst):
                    return semiring.from_int(0)
            else:
                index_to_vembed[index] = vembed
    projected_tensors = [project(tensor.physical, None, tensor.pembeds, subst)
                         for tensor in freshened_tensors]
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

def add(t: EmbeddedTensor, u: EmbeddedTensor) -> EmbeddedTensor:
    """
    Add two EmbeddedTensors. We use anti-unification to compute how much they
    need to be expanded in order to match.
    """
    antisubst : AntiSubst = ({}, {})
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

