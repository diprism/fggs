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
from typing import Sequence, Iterator, List, Tuple, Dict, Set, FrozenSet, Any, Optional, Union, Callable, cast
from dataclasses import dataclass
from abc import ABC, abstractmethod
from warnings import warn
from itertools import zip_longest, chain, count, product, repeat
from functools import reduce
from operator import mul
from math import inf, log, log1p, exp, expm1, isnan, isinf
from string import ascii_uppercase
from sys import float_info#, stderr
import torch
from torch import Tensor, Size
import torch_semiring_einsum
from fggs.semirings import Semiring

__all__ = ['Embedding', 'EmbeddingVar', 'ProductEmbedding', 'SumEmbedding',
           'EmbeddedTensor', 'stack', 'einsum', 'log_viterbi_einsum_forward']

Rename = Dict["EmbeddingVar", "EmbeddingVar"]

Subst = Dict["EmbeddingVar", "Embedding"]

AntiSubst = Tuple[Dict[Tuple["Embedding", "Embedding"], "EmbeddingVar"],
                  Dict["EmbeddingVar", Tuple["Embedding", "Embedding"]]]

NumberType = Union[bool, int, float] # omit the complex type to work around https://github.com/pytorch/pytorch/pull/91345

def make_letterer(alphabet: str = ascii_uppercase) -> Callable[[Any], str]:
    d : Dict[Any, str] = {}
    l = (''.join(chars) for n in count(1)
                        for chars in product(ascii_uppercase, repeat=n))
    def letter(x: Any) -> str:
        s = d.get(x)
        if s is None:
            s = next(l)
            d[x] = s
        return s
    return letter

debugging_letterer = make_letterer()

class Embedding(ABC):
    """An injective mapping from physical indices to virtual indices."""

    @abstractmethod
    def depict(self, letterer: Callable[[EmbeddingVar], str]) -> str:
        pass

    @abstractmethod
    def numel(self) -> int:
        """The virtual indices in the image of this mapping lie in [0,numel())."""
        pass

    @abstractmethod
    def fv(self, subst: Subst) -> Iterator[EmbeddingVar]:
        """Return the set of all EmbeddingVars in self."""
        pass

    @abstractmethod
    def stride(self, subst: Subst) -> Tuple[int, Dict[EmbeddingVar, int]]:
        """The coefficients of the affine map from physical to virtual indices."""
        pass

    def prime_factors(self, subst: Subst) -> Iterator[Embedding]:
        """Express this embedding as a product of non-product embeddings."""
        yield self

    def forward(self, subst: Subst) -> Embedding:
        """Look in subst for the end of the forwarding chain starting with self."""
        return self

    @abstractmethod
    def clone(self, subst: Subst) -> Embedding:
        """Apply the given substitution, creating new Embedding objects except
           reuse existing EmbeddingVar objects."""
        pass

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

    @abstractmethod
    def reassociate(self) -> Iterator[Embedding]:
        """Remove EmbeddingVars of size 1, and merge nested ProductEmbeddings."""
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
                warn(f"Attempt to unify {e.depict(debugging_letterer)} and {f.depict(debugging_letterer)} indicates index type mismatch")
                return False
        if isinstance(e, SumEmbedding) and isinstance(f, SumEmbedding):
            if e.before == f.before and e.after == f.after:
                return e.term.unify(f.term, subst)
            else:
                if __debug__:
                    etn = e.term.numel()
                    ftn = f.term.numel()
                    if e.before + etn + e.after != f.before + ftn + f.after \
                       or e.before + etn > f.before and f.before + ftn > e.before:
                        warn(f"Attempt to unify {e.depict(debugging_letterer)} and {f.depict(debugging_letterer)} indicates index type mismatch")
                return False
        if __debug__:
            if e.numel() != f.numel():
                warn(f"Attempt to unify {e.depict(debugging_letterer)} and {f.depict(debugging_letterer)} indicates index type mismatch")
        if isinstance(e, EmbeddingVar):
            subst[e] = f
            return True
        if isinstance(f, EmbeddingVar):
            subst[f] = e
            return True
        warn(f"Attempt to unify {e.depict(debugging_letterer)} and {f.depict(debugging_letterer)} indicates index type mismatch")
        return False

    def antiunify(e: Embedding, f: Embedding, antisubst: AntiSubst) -> Embedding:
        """Antiunify the two embeddings by extending antisubst.  Return least
           general generalization (whose variables are all in antisubst)."""
        if __debug__:
            if e.numel() != f.numel():
                warn(f"Attempt to antiunify {e.depict(debugging_letterer)} and {f.depict(debugging_letterer)} indicates index type mismatch")
        if isinstance(e, ProductEmbedding) and isinstance(f, ProductEmbedding) and \
           len(e.factors) == len(f.factors):
            return ProductEmbedding(tuple(e1.antiunify(f1, antisubst)
                                          for e1, f1 in zip(e.factors, f.factors)))
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

    def depict(self, letterer):
        return f"{letterer(self)}({self._numel})"

    def numel(self):
        return self._numel

    def fv(self, subst):
        fwd = self.forward(subst)
        if fwd is self:
            yield self
        else:
            yield from fwd.fv(subst)

    def stride(self, subst):
        fwd = self.forward(subst)
        return (0, {self: 1}) if fwd is self else fwd.stride(subst)

    def prime_factors(self, subst):
        fwd = self.forward(subst)
        if fwd is self:
            yield self
        else:
            yield from fwd.prime_factors(subst)

    def forward(self, subst):
        if self in subst:
            subst[self] = ret = subst[self].forward(subst)
            return ret
        else:
            return self

    def clone(self, subst):
        e = subst.get(self, self)
        return self if e is self else e.clone(subst)

    def alpha(e, f, rename):
        return rename.get(e, None) is f

    def freshen(self, rename):
        if self in rename:
            return rename[self]
        else:
            rename[self] = ret = EmbeddingVar(self._numel)
            return ret

    def reassociate(self):
        if self._numel != 1:
            yield self

@dataclass(frozen=True)
class ProductEmbedding(Embedding):
    factors: Tuple[Embedding, ...]

    def depict(self, letterer):
        return f"({'*'.join(e.depict(letterer) for e in self.factors)})"

    def numel(self):
        return reduce(mul, (e.numel() for e in self.factors), 1)

    def fv(self, subst):
        for e in self.factors:
            yield from e.fv(subst)

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

    def prime_factors(self, subst):
        for e in self.factors:
            yield from e.prime_factors(subst)

    def clone(self, subst):
        return ProductEmbedding(tuple(e.clone(subst) for e in self.factors))

    def alpha(e, f, rename):
        return isinstance(f, ProductEmbedding) and \
               len(e.factors) == len(f.factors) and \
               all(e1.alpha(f1, rename) for e1, f1 in zip(e.factors, f.factors))

    def freshen(self, rename):
        return ProductEmbedding(tuple(e.freshen(rename) for e in self.factors))

    def reassociate(self):
        for e in self.factors:
            yield from e.reassociate()

@dataclass(frozen=True)
class SumEmbedding(Embedding):
    before: int
    term: Embedding
    after: int

    def depict(self, letterer) -> str:
        return f"({self.before}+{self.term.depict(letterer)}+{self.after})"

    def numel(self):
        return self.before + self.term.numel() + self.after

    def fv(self, subst):
        yield from self.term.fv(subst)

    def stride(self, subst):
        (o, s) = self.term.stride(subst)
        return (o + self.before, s)

    def clone(self, subst):
        return SumEmbedding(self.before, self.term.clone(subst), self.after)

    def alpha(e, f, rename):
        return isinstance(f, SumEmbedding) and \
               e.before == f.before and e.after == f.after and \
               e.term.alpha(f.term, rename)

    def freshen(self, rename):
        return SumEmbedding(self.before, self.term.freshen(rename), self.after)

    def reassociate(self):
        factors = tuple(self.term.reassociate())
        return SumEmbedding(self.before,
                            factors[0] if len(factors) == 1 else ProductEmbedding(factors),
                            self.after)

def project(virtual: Tensor,
            pembeds: Optional[Sequence[EmbeddingVar]],
            vembeds: Sequence[Embedding],
            subst: Subst) -> Tuple[Tensor, Sequence[EmbeddingVar]]:
    """Extract a view of the given tensor, so that indexing into the returned
       tensor according to pembeds is equivalent to indexing into the given
       tensor according to vembeds."""
    if __debug__:
        if virtual.size() != Size(e.numel() for e in vembeds):
            raise ValueError(f"project(tensor of {virtual.size()}, ..., vembeds of {Size(e.numel() for e in vembeds)}")
    if not subst and pembeds is None \
                 and all(isinstance(e, EmbeddingVar) for e in vembeds) \
                 and len(frozenset(vembeds)) == virtual.ndim:
        # Try to optimize for a common case
        return (virtual, cast(Sequence[EmbeddingVar], vembeds))
    offset = virtual.storage_offset()
    stride : Dict[EmbeddingVar, int] = {}
    for e, n in zip(vembeds, virtual.stride()):
        o, s = e.stride(subst)
        offset += o * n
        for k in s: stride[k] = stride.get(k, 0) + s[k] * n
    if pembeds is None:
        pembeds = tuple(stride.keys())
    else:
        if __debug__:
            vembeds_fv = frozenset(stride.keys())
            pembeds_fv = frozenset(pembeds)
            if vembeds_fv != pembeds_fv:
                raise ValueError(f"project(..., pembeds with {pembeds_fv}, vembeds with {vembeds_fv})")
    return (virtual.as_strided(tuple(k._numel  for k in pembeds),
                               tuple(stride[k] for k in pembeds),
                               offset),
            pembeds)

def reshape_or_view(f: Callable[[Tensor, List[int]], Tensor],
                    self: EmbeddedTensor,
                    *shape: Union[int, Sequence[int]]) -> EmbeddedTensor:
    """Produce a new EmbeddedTensor, with the same elements when flattened,
       whose size() is equal to s."""
    s = list(n for arg in shape for n in ((arg,) if isinstance(arg, int) else arg))
    if self.physical.numel() <= 1:
        return EmbeddedTensor(f(self.physical, s), default=self.default)
    numel = self.numel()
    try:
        inferred = s.index(-1)
    except ValueError:
        inferred = None
    if inferred is not None:
        s[inferred] = numel // reduce(mul, s, -1)
    assert(all(goal >= 0 for goal in s))
    assert(numel == reduce(mul, s, 1))
    subst : Subst = {}
    primes : Iterator[Embedding] = (prime for e in self.vembeds
                                          for prime in e.prime_factors(subst))
    packs : List[List[Embedding]] = []
    for goal in s:
        pack = []
        while goal != 1:
            try:
                prime = next(primes)
            except StopIteration:
                raise RuntimeError(f'Cannot reshape_or_view {self} to {s}')
            pn = prime.numel()
            if pn > goal:
                if isinstance(prime, EmbeddingVar) and 0 == pn % goal:
                    p1 = EmbeddingVar(goal)
                    p2 = EmbeddingVar(pn // goal)
                    subst[prime] = ProductEmbedding((p1, p2))
                    prime = p1
                    pn = goal
                    primes = chain((p2,), primes)
                else:
                    raise RuntimeError(f'Cannot reshape_or_view {self} to {s}')
            if 0 != goal % pn:
                raise RuntimeError(f'Cannot reshape_or_view {self} to {s}')
            pack.append(prime)
            goal //= pn
        packs.append(pack)
    try:
        while True:
            packs[-1].append(next(primes))
    except StopIteration:
        pass
    pembeds = tuple(k for e in self.pembeds
                      for k in e.prime_factors(subst))
    vembeds = tuple(pack[0].clone(subst) if len(pack) == 1
                    else ProductEmbedding(tuple(e.clone(subst) for e in pack))
                    for pack in packs)
    assert(len(vembeds) == len(s) and
           all(e.numel() == goal for e, goal in zip(vembeds, s)))
    return EmbeddedTensor(f(self.physical, list(k._numel for k in pembeds)),
                          pembeds, vembeds, self.default).reassociate()

def broadcast(vembedss: Sequence[Sequence[Embedding]]) \
    -> Tuple[Tuple[List[Embedding], List[EmbeddingVar]], ...]:
    """Apply broadcast semantics to the given vembed sequences
       to make them equal in length (maximum ndim) and size,
       creating fresh EmbeddingVars for each broadcast dimension.
       We also return lists of the fresh EmbeddingVars created."""
    rets : Tuple[Tuple[List[Embedding], List[EmbeddingVar]], ...] \
         = tuple(([], []) for vembed in vembedss)
    for es in zip_longest(*(reversed(vembeds) for vembeds in vembedss),
                          fillvalue=ProductEmbedding(())):
        ns = tuple(e.numel() for e in es)
        Ns = frozenset(n for n in ns if n != 1)
        if len(Ns) > 1: raise RuntimeError(f"Size mismatch {ns}")
        N = tuple(Ns)[0] if Ns else 1
        for ret, e, n in zip(rets, es, ns):
            if n != N:
                k = EmbeddingVar(N)
                e = k if e == ProductEmbedding(()) else ProductEmbedding((e, k))
                ret[1].insert(0, k)
            ret[0].insert(0, e)
    return rets

@dataclass
class EmbeddedTensor:
    physical: Tensor
    pembeds: Sequence[EmbeddingVar] = cast(Sequence[EmbeddingVar], None) #-/
    vembeds: Sequence[Embedding]    = cast(Sequence[Embedding],    None) #-/
    default: NumberType = 0

    def __post_init__(self):
        # Default to the trivial (dense) embedding, to convert from Tensor
        if self.pembeds is None:
            self.pembeds = tuple(EmbeddingVar(n) for n in self.physical.size())
        else:
            if __debug__:
                if self.physical.size() != Size(k.numel() for k in self.pembeds):
                    raise ValueError(f"EmbeddedTensor(tensor of {self.physical.size()}, pembeds of {Size(k.numel() for k in self.pembeds)}, ...)")
        if self.vembeds is None:
            self.vembeds = self.pembeds

    @staticmethod
    def from_int(x: int, semiring: Semiring) -> EmbeddedTensor:
        return EmbeddedTensor(semiring.from_int(x), default=semiring.from_int(0).item())

    @staticmethod
    def eye(size: int, semiring: Semiring) -> EmbeddedTensor:
        k = EmbeddingVar(size)
        return EmbeddedTensor(semiring.from_int(1).expand(size), (k,), (k,k),
                              semiring.from_int(0).item())

    def depict(self, letterer: Callable[[EmbeddingVar], str]) -> str:
        P = (k.depict(letterer) for k in self.pembeds)
        V = (e.depict(letterer) for e in self.vembeds)
        return f"[\\{' '.join(P)} -> {', '.join(V)} | default {self.default}]"

    @property
    def shape(self):
        return self.size()

    def numel(self) -> int:
        return reduce(mul, (e.numel() for e in self.vembeds), 1)

    def size(self) -> Size:
        return Size(e.numel() for e in self.vembeds)

    def dim(self) -> int:
        return len(self.vembeds)

    ndim = ndimension = property(dim)

    @property
    def dtype(self) -> torch.dtype:
        return self.physical.dtype

    @property
    def requires_grad(self) -> bool:
        return self.physical.requires_grad

    def is_complex(self) -> bool:
        return self.physical.is_complex()

    def freshen(self) -> EmbeddedTensor:
        """Return a new EmbeddedTensor (with same underlying physical storage)
           with fresh EmbeddingVars."""
        rename : Rename = {}
        return EmbeddedTensor(self.physical,
                              tuple(k.freshen(rename) for k in self.pembeds),
                              tuple(e.freshen(rename) for e in self.vembeds),
                              self.default)

    def reassociate(self) -> EmbeddedTensor:
        """Remove physical dimensions of size 1, and merge nested ProductEmbeddings."""
        return EmbeddedTensor(self.physical.squeeze(),
                              tuple(k for k in self.pembeds if k._numel != 1),
                              tuple(factors[0] if len(factors) == 1 else ProductEmbedding(factors)
                                    for e in self.vembeds
                                    for factors in (tuple(e.reassociate()),)),
                              self.default)

    def default_to(self, default: NumberType) -> EmbeddedTensor:
        return self if self.default == default or isnan(self.default) and isnan(default) \
               else EmbeddedTensor(self.to_dense(), default=default)

    def clone(self) -> EmbeddedTensor:
        rename : Rename = {}
        return EmbeddedTensor(self.physical.clone(),
                              tuple(k.freshen(rename) for k in self.pembeds),
                              tuple(e.freshen(rename) for e in self.vembeds),
                              self.default)

    def detach(self: EmbeddedTensor) -> EmbeddedTensor:
        rename : Rename = {}
        return EmbeddedTensor(self.physical.detach(),
                              tuple(k.freshen(rename) for k in self.pembeds),
                              tuple(e.freshen(rename) for e in self.vembeds),
                              self.default)

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
        self.default = src.default

    def isdisjoint(self, other: Union[Set[EmbeddingVar], EmbeddedTensor]) -> bool:
        """Check whether the EmbeddingVars in self and other overlap."""
        return (frozenset(other.pembeds) if isinstance(other, EmbeddedTensor) else other) \
               .isdisjoint(self.pembeds)

    def to_dense(self) -> Tensor:
        """Expand a physical tensor to a mostly-zero virtual tensor."""
        if len(self.pembeds) == len(self.vembeds) and \
           all(isinstance(e, EmbeddingVar) for e in self.vembeds) and \
           frozenset(self.pembeds) == frozenset(self.vembeds):
            # vembeds is just a permutation of pembeds, so just clone view on physical
            return project(self.physical,
                           cast(Tuple[EmbeddingVar], self.vembeds),
                           self.pembeds, {})[0].clone()
        virtual = self.physical.new_full(self.size(), self.default)
        project(virtual, self.pembeds, self.vembeds, {})[0].copy_(self.physical)
        return virtual

    def masked_fill_into(self, dest: Tensor, value: NumberType) -> None:
        """Fill elements of dest tensor with value where self is True."""
        projected = project(dest, self.pembeds, self.vembeds, {})[0]
        if self.default:
            preserved = torch.where(self.physical, value, projected)
            dest.fill_(value)
            projected.copy_(preserved)
        else:
            projected.masked_fill_(self.physical, value)

    def dim_to_dense(self, dim: int) -> EmbeddedTensor:
        """Expand to an equivalent EmbeddedTensor but make sure that the given
           dimension is dense and independent of the other dimensions."""
        vembeds = list(self.vembeds)
        e_dense = vembeds.pop(dim)
        rename : Rename = {}
        vembeds = list(e.freshen(rename) for e in vembeds)
        if isinstance(e_dense, EmbeddingVar) and e_dense not in rename:
            return self
        fv  = tuple(rename.keys())
        fv0 = tuple(rename.values())
        k = EmbeddingVar(e_dense.numel())
        vembeds.insert(dim, k)
        pembeds = fv0 + (k,)
        default = self.default
        physical = self.physical.new_full(tuple(e._numel for e in pembeds), default)
        project(physical, self.pembeds, fv + (e_dense,), {})[0].copy_(self.physical)
        return EmbeddedTensor(physical, pembeds, vembeds, default)

    def any(self, dim: int, keepdim: bool = False) -> EmbeddedTensor:
        vembeds = list(self.vembeds)
        if keepdim:
            vembeds[dim] = ProductEmbedding(())
        else:
            del vembeds[dim]
        ks = frozenset(self.vembeds[dim].fv({})) - frozenset(k for e in vembeds for k in e.fv({}))
        pembeds = tuple(k for k in self.pembeds if k not in ks)
        if self.default and self.vembeds[dim].numel() > reduce(mul, (k._numel for k in ks), 1):
            physical = self.physical.new_ones(()).expand(tuple(k._numel for k in pembeds))
        else:
            physical = self.physical
            for i in range(len(self.pembeds)-1, -1, -1):
                if self.pembeds[i] in ks:
                    physical = physical.any(dim=i, keepdim=False)
        return EmbeddedTensor(physical, pembeds, vembeds, self.default)

    def permute(self, dims: Sequence[int]) -> EmbeddedTensor:
        assert(len(dims) == len(self.vembeds))
        assert(frozenset(dims) == frozenset(range(0,len(dims))))
        return EmbeddedTensor(self.physical,
                              self.pembeds,
                              tuple(self.vembeds[i] for i in dims),
                              self.default)

    def __iter__(self) -> Iterator[EmbeddedTensor]:
        self = self.dim_to_dense(0)
        vembeds = list(self.vembeds)
        k = vembeds.pop(0)
        assert(isinstance(k, EmbeddingVar))
        pembeds = list(self.pembeds)
        i = pembeds.index(k)
        pembeds.pop(i)
        perm = (i, *range(0, i), *range(i+1, len(self.pembeds)))
        for p in self.physical.permute(perm):
            yield EmbeddedTensor(p, pembeds, vembeds, self.default)

    def neg_(self) -> EmbeddedTensor:
        self.default = -self.default
        self.physical.neg_()
        return self

    def log_(self) -> EmbeddedTensor:
        self.default = log(self.default) if self.default else -inf
        self.physical.log_()
        return self

    def log1p_(self) -> EmbeddedTensor:
        self.default = log1p(self.default)
        self.physical.log1p_()
        return self

    def relu_(self) -> EmbeddedTensor:
        self.default = max(0, self.default)
        self.physical.relu_()
        return self

    def abs_(self) -> EmbeddedTensor:
        self.default = abs(self.default)
        self.physical.abs_()
        return self

    def nan_to_num_(self, nan: float = 0.,
                          posinf: Optional[float] = None,
                          neginf: Optional[float] = None) -> EmbeddedTensor:
        if isnan(self.default):
            self.default = nan
        elif isinf(self.default):
            if self.default >= 0:
                self.default = float_info.max if posinf is None else posinf
            else:
                self.default = -float_info.max if neginf is None else neginf
        self.physical.nan_to_num_(nan=nan, posinf=posinf)
        return self

    def __imul__(self, other: NumberType) -> EmbeddedTensor:
        self.default  *= other
        self.physical *= other
        return self

    def __itruediv__(self, other: NumberType) -> EmbeddedTensor:
        self.default  /= other
        self.physical /= other
        return self

    def lt(self, other: Union[float, EmbeddedTensor]) -> EmbeddedTensor:
        if isinstance(other, EmbeddedTensor):
            return self.binary(other, self.default < other.default, torch.lt)
        else:
            return EmbeddedTensor(self.physical.lt(other), self.pembeds, self.vembeds, self.default < other)

    def le(self, other: Union[float, EmbeddedTensor]) -> EmbeddedTensor:
        if isinstance(other, EmbeddedTensor):
            return self.binary(other, self.default <= other.default, torch.le)
        else:
            return EmbeddedTensor(self.physical.le(other), self.pembeds, self.vembeds, self.default <= other)

    def gt(self, other: Union[float, EmbeddedTensor]) -> EmbeddedTensor:
        if isinstance(other, EmbeddedTensor):
            return self.binary(other, self.default > other.default, torch.gt)
        else:
            return EmbeddedTensor(self.physical.gt(other), self.pembeds, self.vembeds, self.default > other)

    def ge(self, other: Union[float, EmbeddedTensor]) -> EmbeddedTensor:
        if isinstance(other, EmbeddedTensor):
            return self.binary(other, self.default >= other.default, torch.ge)
        else:
            return EmbeddedTensor(self.physical.ge(other), self.pembeds, self.vembeds, self.default >= other)

    def eq(self, other: Union[float, EmbeddedTensor]) -> EmbeddedTensor:
        if isinstance(other, EmbeddedTensor):
            return self.binary(other, self.default == other.default, torch.eq)
        else:
            return EmbeddedTensor(self.physical.eq(other), self.pembeds, self.vembeds, self.default == other)

    def to(self, dtype: torch.dtype) -> EmbeddedTensor:
        return EmbeddedTensor(self.physical.to(dtype=dtype), self.pembeds, self.vembeds,
                              self.physical.new_tensor(self.default, dtype=dtype).item())

    def abs(self) -> EmbeddedTensor:
        return EmbeddedTensor(self.physical.abs(), self.pembeds, self.vembeds, abs(self.default))

    def exp(self) -> EmbeddedTensor:
        return EmbeddedTensor(self.physical.exp(), self.pembeds, self.vembeds, exp(self.default))

    def expm1(self) -> EmbeddedTensor:
        return EmbeddedTensor(self.physical.expm1(), self.pembeds, self.vembeds, expm1(self.default))

    def log(self) -> EmbeddedTensor:
        return EmbeddedTensor(self.physical.log(), self.pembeds, self.vembeds,
                              log(self.default) if self.default else -inf)

    def logical_not(self) -> EmbeddedTensor:
        return EmbeddedTensor(~self.physical, self.pembeds, self.vembeds, not self.default)

    def __mul__(self, other: NumberType) -> EmbeddedTensor:
        return EmbeddedTensor(self.physical * other, self.pembeds, self.vembeds, self.default * other)

    def __truediv__(self, other: NumberType) -> EmbeddedTensor:
        return EmbeddedTensor(self.physical / other, self.pembeds, self.vembeds, self.default / other)

    def log_softmax(self, dim) -> EmbeddedTensor:
        if dim < 0: dim += self.ndim
        self = self.dim_to_dense(dim)
        k = self.vembeds[dim]
        i = self.pembeds.index(k)
        return EmbeddedTensor(self.physical.log_softmax(dim = i),
                              self.pembeds, self.vembeds, -log(k._numel))

    def equal_default(self) -> bool:
        return cast(bool, self.physical.eq(self.default).all().item())

    def allclose_default(self, rtol=1e-05, atol=1e-08) -> bool:
        return self.physical.allclose(self.physical.new_tensor(self.default),
                                      rtol=rtol, atol=atol, equal_nan=True)

    def equal(self, other: EmbeddedTensor) -> bool:
        s = self.size()
        if s != other.size(): return False
        n = s.numel()
        if not self.isdisjoint(other): other = other.freshen()
        selfok  = self.physical == other.default
        otherok = other.physical == self.default
        # Compare overlapping parts to each other
        subst : Subst = {}
        if all(v.unify(u, subst) for v, u in zip(self.vembeds, other.vembeds)):
            # There is overlap
            subself, subembeds = project(self.physical, None, self.pembeds, subst)
            subother, _ = project(other.physical, subembeds, other.pembeds, subst)
            if not subself.equal(subother): return False
            project(selfok , None, self .pembeds, subst)[0].fill_(True)
            project(otherok, None, other.pembeds, subst)[0].fill_(True)
            n += subself.numel()
        # Compare defaults and non-overlapping parts
        return (n <= selfok.numel() + otherok.numel() or
                self.default == other.default) and \
               bool(selfok.all()) and bool(otherok.all())

    def allclose(self, other: EmbeddedTensor, rtol=1e-05, atol=1e-08, equal_nan=False) -> bool:
        s = self.size()
        if s != other.size(): return False
        n = s.numel()
        if not self.isdisjoint(other): other = other.freshen()
        selfok = self.physical.isclose(other.physical.new_tensor(other.default), rtol=rtol, atol=atol, equal_nan=equal_nan)
        otherok = self.physical.new_tensor(self.default).isclose(other.physical, rtol=rtol, atol=atol, equal_nan=equal_nan)
        # Compare overlapping parts to each other
        subst : Subst = {}
        if all(v.unify(u, subst) for v, u in zip(self.vembeds, other.vembeds)):
            # There is overlap
            subself, subembeds = project(self.physical, None, self.pembeds, subst)
            subother, _ = project(other.physical, subembeds, other.pembeds, subst)
            if not subself.allclose(subother, rtol=rtol, atol=atol, equal_nan=equal_nan): return False
            project(selfok , None, self .pembeds, subst)[0].fill_(True)
            project(otherok, None, other.pembeds, subst)[0].fill_(True)
            n += subself.numel()
        # Compare defaults and non-overlapping parts
        return (n <= selfok.numel() + otherok.numel() or
                self.physical.new_tensor(self.default)
                    .allclose(other.physical.new_tensor(other.default),
                              rtol=rtol, atol=atol, equal_nan=equal_nan)) and \
               bool(selfok.all()) and bool(otherok.all())

    def expansion(t, u: EmbeddedTensor) -> Tuple[Sequence[EmbeddingVar],
                                                 Sequence[Embedding],
                                                 Sequence[EmbeddingVar],
                                                 Sequence[Embedding],
                                                 Sequence[EmbeddingVar],
                                                 Sequence[Embedding]]:
        """Compute how to expand two EmbeddedTensors in order to match."""
        antisubst : AntiSubst = ({}, {})
        pembeds1 : List[EmbeddingVar] = list(t.pembeds)
        pembeds2 : List[EmbeddingVar] = list(u.pembeds)
        lggs : List[Embedding] = []
        for e, f in zip_longest(reversed(t.vembeds),
                                reversed(u.vembeds),
                                fillvalue=ProductEmbedding(())):
            if e == ProductEmbedding(()) != f:
                new = EmbeddingVar(f.numel())
                pembeds1.insert(0, new)
                antisubst[0][(new, f)] = new
                antisubst[1][new] = (new, f)
                lggs.insert(0, new)
            elif f == ProductEmbedding(()) != e:
                new = EmbeddingVar(e.numel())
                pembeds2.insert(0, new)
                antisubst[0][(e, new)] = new
                antisubst[1][new] = (e, new)
                lggs.insert(0, new)
            else:
                lggs.insert(0, e.antiunify(f, antisubst))
        (gs, es, fs) = zip(*((g, e, f) for (g, (e, f)) in antisubst[1].items())) \
                       if antisubst[1] else ((), (), ())
        return (gs, lggs, pembeds1, es, pembeds2, fs)

    def commutative(t, u: EmbeddedTensor, identity: NumberType, default: NumberType,
                    operate_: Callable[[Tensor, Tensor], Any]) -> EmbeddedTensor:
        """Apply a symmetric binary operation with identity."""
        (gs, lggs, pembeds1, es, pembeds2, fs) = t.expansion(u)
        tp = t.physical.expand(tuple(k._numel for k in pembeds1))
        up = u.physical.expand(tuple(k._numel for k in pembeds2))
        if t.default != identity or \
           len(pembeds1) != len(t.pembeds) or \
           len(pembeds2) == len(u.pembeds) and t.physical.numel() >= u.physical.numel():
            td = EmbeddedTensor(tp, pembeds1, es, t.default).to_dense()
            if u.default == identity:
                tp = project(td, pembeds2, fs, {})[0]
                operate_(tp, up)
            else:
                ud = EmbeddedTensor(up, pembeds2, fs, u.default).to_dense()
                operate_(td, ud)
            return EmbeddedTensor(td, gs, lggs, default)
        else:
            ud = EmbeddedTensor(up, pembeds2, fs, u.default).to_dense()
            up = project(ud, pembeds1, es, {})[0]
            operate_(up, tp)
            return EmbeddedTensor(ud, gs, lggs, default)

    def binary(t, u: EmbeddedTensor, default: NumberType,
               operate: Callable[[Tensor, Tensor], Tensor]) -> EmbeddedTensor:
        """Apply a binary operation."""
        (gs, lggs, pembeds1, es, pembeds2, fs) = t.expansion(u)
        tp = t.physical.expand(tuple(k._numel for k in pembeds1))
        up = u.physical.expand(tuple(k._numel for k in pembeds2))
        td = EmbeddedTensor(tp, pembeds1, es, t.default).to_dense()
        ud = EmbeddedTensor(up, pembeds2, fs, u.default).to_dense()
        return EmbeddedTensor(operate(td, ud), gs, lggs, default)

    def add(t, u: EmbeddedTensor) -> EmbeddedTensor:
        return t.commutative(u, 0, t.default + u.default,
                             lambda x, y: x.add_(y))

    def mul(t, u: EmbeddedTensor) -> EmbeddedTensor:
        return t.commutative(u, 1, t.default * u.default,
                             lambda x, y: x.mul_(y))

    def logaddexp(t, u: EmbeddedTensor) -> EmbeddedTensor:
        return t.commutative(u, -inf, t.physical.new_tensor(t.default).logaddexp(
                                      u.physical.new_tensor(u.default)).item(),
                             lambda x, y: torch.logaddexp(x, y, out=x))

    def maximum(t, u: EmbeddedTensor) -> EmbeddedTensor:
        return t.commutative(u, -inf, max(t.default, u.default),
                             lambda x, y: torch.maximum(x, y, out=x))

    def logical_or(t, u: EmbeddedTensor) -> EmbeddedTensor:
        # TODO: short circuiting
        return t.commutative(u, False, t.default or u.default,
                             lambda x, y: x.logical_or_(y))

    def logical_and(t, u: EmbeddedTensor) -> EmbeddedTensor:
        # TODO: short circuiting
        return t.commutative(u, True, t.default and u.default,
                             lambda x, y: x.logical_and_(y))

    def sub(t, u: EmbeddedTensor) -> EmbeddedTensor:
        """Subtract two EmbeddedTensors."""
        (gs, lggs, pembeds1, es, pembeds2, fs) = t.expansion(u)
        tp = t.physical.expand(tuple(k._numel for k in pembeds1))
        default = t.default - u.default
        if t.default != 0 or \
           len(pembeds1) != len(t.pembeds) or \
           len(pembeds2) == len(u.pembeds) and t.physical.numel() >= u.physical.numel():
            up = u.physical.expand(tuple(k._numel for k in pembeds2))
            td = EmbeddedTensor(tp, pembeds1, es, t.default).to_dense()
            if u.default == 0:
                tp = project(td, pembeds2, fs, {})[0]
                tp.sub_(up)
            else:
                ud = EmbeddedTensor(up, pembeds2, fs, u.default).to_dense()
                td.sub_(ud)
            return EmbeddedTensor(td, gs, lggs, default)
        else:
            up = (-u.physical).expand(tuple(k._numel for k in pembeds2))
            ud = EmbeddedTensor(up, pembeds2, fs, -u.default).to_dense()
            up = project(ud, pembeds1, es, {})[0]
            up.add_(tp)
            return EmbeddedTensor(ud, gs, lggs, default)

    __add__ = add
    __sub__ = sub

    def where(t, c, u) -> EmbeddedTensor:
        if c.default: t, u = u, t
        if not t.isdisjoint(c): t = t.freshen()
        # Broadcast
        (t_vembeds, _), (c_vembeds, c_new_pembeds), (u_vembeds, u_new_pembeds) \
            = broadcast((t.vembeds, c.vembeds, u.vembeds))
        c_pembeds = tuple(chain(c_new_pembeds, c.pembeds))
        u_pembeds = tuple(chain(u_new_pembeds, u.pembeds))

        # Extract the intersection of the non-default regions of t and c,
        # and match up their dimensions
        subst : Subst = {}
        success = all(ec.unify(et, subst) for ec, et in zip(c_vembeds, t_vembeds))
        if success:
            kst    : FrozenSet[EmbeddingVar] = frozenset(k for e in t.pembeds     for k in e.fv(subst))
            ksc    : FrozenSet[EmbeddingVar] = frozenset(k for e in c.pembeds     for k in e.fv(subst))
            ksc_new: FrozenSet[EmbeddingVar] = frozenset(k for e in c_new_pembeds for k in e.fv(subst))
            ks     : Tuple[EmbeddingVar,...] = tuple(ksc | ksc_new)
            projected_t = project(t.physical, tuple(k for k in ks if k in kst), t.pembeds, subst)[0]
            for i, k in enumerate(ks):
                if k not in kst: projected_t = projected_t.unsqueeze(i)
            projected_c = project(c.physical, tuple(k for k in ks if k in ksc), c.pembeds, subst)[0]
            for i, k in enumerate(ks):
                if k not in ksc: projected_c = projected_c.unsqueeze(i)
            assert(projected_t.ndim == projected_c.ndim and
                   all(len({n1, n2} - {1}) <= 1 for n1, n2 in zip(projected_t.size(),
                                                                  projected_c.size())))

            # Is the non-default region of t a superset of the non-default region of c?
            # In other words, did unification do anything to c_pembeds?
            fullness_test = frozenset(e.forward(subst) for e in c_pembeds)
            full = len(fullness_test) == len(c_pembeds) \
                   and all(isinstance(e, EmbeddingVar) for e in fullness_test)

        # Expand u to c
        antisubst : AntiSubst = ({}, {})
        lggs = [ec.antiunify(eu, antisubst)
                for ec, eu in zip(c_vembeds, u_vembeds)]
        (gs, ecs, eus) = zip(*((g, ec, eu) for (g, (ec, eu)) in antisubst[1].items())) \
                         if antisubst[1] else ((), (), ())
        up = u.physical.expand(tuple(k._numel for k in u_pembeds))
        ud = EmbeddedTensor(up, u_pembeds, eus, u.default).to_dense()

        if not (success and full):
            # Fill expanded u with t.default
            up = project(ud, c_pembeds, ecs, {})[0]
            up.masked_fill_(~c.physical if c.default else c.physical, t.default)
        if success:
            # Mutate expanded u with matched t
            up = project(ud, ks, ecs, subst)[0]
            up.copy_(up.where(projected_c, projected_t) if c.default
                     else projected_t.where(projected_c, up))

        return EmbeddedTensor(ud, gs, lggs, u.default)

    def transpose(self, dim0, dim1) -> EmbeddedTensor:
        if dim0 == dim1:
            return self
        else:
            if dim0 > dim1: dim0, dim1 = dim1, dim0
            vembeds = self.vembeds
            assert(0 <= dim0 < dim1 < len(vembeds))
            vembeds = (vembeds[      :dim0  ] + # type: ignore
                       vembeds[dim1  :dim1+1] +
                       vembeds[dim0+1:dim1  ] +
                       vembeds[dim0  :dim0+1] +
                       vembeds[dim1+1:      ])
            return EmbeddedTensor(self.physical, self.pembeds, vembeds, self.default)

    def t(self) -> EmbeddedTensor:
        assert(len(self.vembeds) <= 2)
        if len(self.vembeds) < 2: return self
        return self.transpose(0,1)

    @property
    def T(self) -> EmbeddedTensor:
        return EmbeddedTensor(self.physical, self.pembeds, self.vembeds[::-1], self.default)

    def flatten(self) -> EmbeddedTensor:
        if len(self.vembeds) == 1:
            return self.reassociate()
        else:
            return EmbeddedTensor(self.physical,
                                  self.pembeds,
                                  (ProductEmbedding(tuple(self.vembeds)),),
                                  self.default).reassociate()

    def unsqueeze(self, dim: int) -> EmbeddedTensor:
        if dim < 0: dim += self.ndim + 1
        vembeds = list(self.vembeds)
        vembeds.insert(dim, ProductEmbedding(()))
        return EmbeddedTensor(self.physical, self.pembeds, vembeds, self.default)

    def reshape(self, *shape: Union[int, Sequence[int]]) -> EmbeddedTensor:
        return reshape_or_view(lambda tensor, size: tensor.reshape(size), self, *shape)

    def view(self, *shape: Union[int, Sequence[int]]) -> EmbeddedTensor:
        return reshape_or_view(lambda tensor, size: tensor.view(size), self, *shape)

    def expand(self: EmbeddedTensor, *sizes: int) -> EmbeddedTensor:
        pembeds: List[EmbeddingVar] = list(self.pembeds)
        vembeds: List[Embedding   ] = []
        for e, n in zip_longest(reversed(self.vembeds), reversed(sizes)):
            if n is None:
                raise RuntimeError(f"EmbeddedTensor.expand: the number of sizes provided ({len(sizes)}) must be greater or equal to the number of dimensions in the tensor ({len(self.vembeds)})")
            if e is None or e.numel() == 1 != n:
                k = EmbeddingVar(n)
                e = k if e is None or e == ProductEmbedding(()) else ProductEmbedding((e, k))
                pembeds.insert(0, k)
            if e.numel() != n:
                raise RuntimeError(f"EmbeddedTensor.expand: cannot extend {self.size()} to {sizes}")
            vembeds.insert(0, e)
        return EmbeddedTensor(self.physical.expand(tuple(k._numel for k in pembeds)),
                              pembeds, vembeds, self.default)

    def repeat(self: EmbeddedTensor, *sizes: int) -> EmbeddedTensor:
        return self.expand(*sizes).clone()

    def solve(a, b: EmbeddedTensor, semiring: Semiring) -> EmbeddedTensor:
        """Solve x = a @ x + b for x."""
        assert(len(a.vembeds) == 2 and len(b.vembeds) >= 1)
        assert(a.vembeds[0].numel() == a.vembeds[1].numel() == b.vembeds[0].numel())
        zero = semiring.from_int(0)
        a = a.default_to(zero.item())
        b = b.default_to(zero.item())
        if not a.isdisjoint(b): b = b.freshen()
        e = b.vembeds[0] # Embedding for b + a*b + ... + a^n*b (initially n=0)
                         # Invariant: a and e are disjoint
        while True: # Compute least embedding for solution. TODO any easier way?
            # Update e := a*e + b
            subst : Subst = {}
            if e.unify(a.vembeds[1], subst):
                antisubst : AntiSubst = ({}, {})
                e = e.antiunify(a.vembeds[0].clone(subst), antisubst)
                if all(isinstance(e1, EmbeddingVar) for e1, _ in antisubst[0]):
                    break # acquiescence
            else:
                break # a*e = 0
        # Copy e  (for naming the columns of a and rows of b)
        #   to e0 (for naming the rows of a)
        rename : Rename = {}
        e0  = e.freshen(rename)
        fv  = tuple(rename.keys())
        fv0 = tuple(rename.values())
        # Copy ProductEmbedding(b.vembeds[1:])
        #   to e1 (for naming the columns of b)
        rename = {}
        ebs0 = tuple(eb.freshen(rename) for eb in b.vembeds[1:])
        fvb  = tuple(rename.keys())
        fvb0 = tuple(rename.values())
        # Convert b to regular matrix tensor
        subst = {}
        if not e.unify(b.vembeds[0], subst): raise AssertionError
        projected_b = project(b.physical, None, b.pembeds, subst)
        dense_b = EmbeddedTensor(projected_b[0],
                                 projected_b[1],
                                 (ProductEmbedding(fv ).clone(subst),
                                  ProductEmbedding(fvb).clone(subst)),
                                 b.default).to_dense()
        # Convert relevant portion of a to regular matrix tensor
        subst = {}
        if not (e0.unify(a.vembeds[0], subst) and
                e .unify(a.vembeds[1], subst)): raise AssertionError
        projected_a = project(a.physical, None, a.pembeds, subst)
        dense_a = EmbeddedTensor(projected_a[0],
                                 projected_a[1],
                                 (ProductEmbedding(fv0).clone(subst),
                                  ProductEmbedding(fv ).clone(subst)),
                                 a.default).to_dense()
        # Solve
        x = semiring.solve(dense_a, dense_b)
        return EmbeddedTensor(x.reshape(tuple(k._numel for k in fv + fvb0)),
                              fv + fvb0, (e,) + ebs0, b.default)

    def mv(self, v: EmbeddedTensor, semiring: Semiring) -> EmbeddedTensor:
        return einsum((self,v), ("ij", "j"), "i", semiring)

    def mm(self, m: EmbeddedTensor, semiring: Semiring) -> EmbeddedTensor:
        return einsum((self,m), ("ij", "jk"), "ik", semiring)

def stack(tensors: Sequence[EmbeddedTensor], dim: int = 0) -> EmbeddedTensor:
    """Concatenate EmbeddedTensors along a new dimension.
       The inputs must have the same (virtual) size and default."""
    assert(tensors)
    head, *tail = tensors
    size = head.size()
    default = head.default
    lggs = head.vembeds
    if not tail:
        lggs = list(lggs)
        lggs.insert(dim, ProductEmbedding(()))
        return EmbeddedTensor(head.physical, head.pembeds, lggs, default)
    for t in tail: # Antiunify all input vembeds
        assert(size == t.size())
        assert(default == t.default)
        antisubst : AntiSubst = ({}, {})
        lggs = tuple(e.antiunify(f, antisubst)
                     for e, f in zip(lggs, t.vembeds))
    k = EmbeddingVar(len(tensors))
    ks = tuple(antisubst[1])
    pembeds = list(ks)
    pembeds.insert(0, k)
    vembeds = list(lggs)
    vembeds.insert(dim, k)
    physical = head.physical.new_full(tuple(e._numel for e in pembeds), default)
    for t, p in zip(tensors, physical):
        subst : Subst = {}
        if not all(e.unify(f, subst)
                   for e, f in zip(lggs, t.vembeds)):
            raise AssertionError
        project(p, tuple(e.forward(subst) for e in t.pembeds), ks, subst)[0].copy_(t.physical)
    return EmbeddedTensor(physical, pembeds, vembeds, default)

def depict_einsum(fun: str,
                  tensors: Sequence[EmbeddedTensor],
                  inputs: Sequence[Sequence[Any]],
                  output: Sequence[Any]) -> str:
    letterer = make_letterer()
    return "\n".join([fun + "("] +
                     [f"    {' '.join(map(letterer, input))} {tensor.depict(letterer)}"
                      for tensor, input in zip(tensors, inputs)] +
                     [f"    -> {' '.join(map(letterer, output))})"])

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
    assert(len(tensors) == len(inputs))
    assert(len(tensor.vembeds) == len(input)
           for tensor, input in zip(tensors, inputs))
    assert(frozenset(index for input in inputs for index in input) >= frozenset(output))
    zero = semiring.from_int(0)
    one  = semiring.from_int(1)
    if len(tensors) == 0:
        return EmbeddedTensor(one, default=zero.item())
    #print(depict_einsum('einsum', tensors, inputs, output), file=stderr)
    tensors = [tensor.default_to(zero.item()) for tensor in tensors]
    pembeds_fv : Set[EmbeddingVar] = set()
    index_to_vembed : Dict[Any, Embedding] = {}
    subst : Subst = {}
    freshened_tensors = []
    result_is_zero = False
    for (i, (tensor, input)) in enumerate(zip(tensors, inputs)):
        if not tensor.isdisjoint(pembeds_fv): tensor = tensor.freshen()
        freshened_tensors.append(tensor)
        pembeds_fv.update(tensor.pembeds)
        for vembed, index in zip(tensor.vembeds, input):
            if index in index_to_vembed:
                if not index_to_vembed[index].unify(vembed, subst):
                    result_is_zero = True
            else:
                index_to_vembed[index] = vembed
    output_vembeds = tuple(index_to_vembed[index].clone(subst) for index in output)
    if result_is_zero:
        # TODO: represent all-zero tensor with empty physical?
        outsize = tuple(e.numel() for e in output_vembeds)
        return EmbeddedTensor(zero.expand(outsize), default=zero.item())
    projected_tensors = [project(tensor.physical, None, tensor.pembeds, subst)
                         for tensor in freshened_tensors]
    pembed_to_char = dict(zip(chain.from_iterable(pembeds for (view, pembeds) in projected_tensors),
                              map(chr, count(ord('a')))))
    output_pembeds = tuple(frozenset(k for e in output_vembeds for k in e.fv(subst)))
    equation = ','.join(''.join(pembed_to_char[k] for k in pembeds)
                        for (view, pembeds) in projected_tensors) \
             + '->' + ''.join(pembed_to_char[k] for k in output_pembeds)
    #print(equation, file=stderr)
    compiled = torch_semiring_einsum.compile_equation(equation)
    out = semiring.einsum(compiled, *(view for (view, pembed) in projected_tensors))
    return EmbeddedTensor(out, output_pembeds, output_vembeds, default=zero.item())

def log_viterbi_einsum_forward(tensors: Sequence[EmbeddedTensor],
                               inputs: Sequence[Sequence[Any]],
                               output: Sequence[Any],
                               semiring: Semiring) -> Tuple[EmbeddedTensor, EmbeddedTensor]:
    assert(len(tensors) == len(inputs))
    assert(len(tensor.vembeds) == len(input)
           for tensor, input in zip(tensors, inputs))
    assert(frozenset(index for input in inputs for index in input) >= frozenset(output))
    zero = semiring.from_int(0)
    one  = semiring.from_int(1)
    if len(tensors) == 0:
        return (EmbeddedTensor(one, default=zero.item()),
                EmbeddedTensor(one.new_empty((0,), dtype=torch.long), default=0))
    #print(depict_einsum('log_viterbi_einsum_forward', tensors, inputs, output), file=stderr)
    tensors = [tensor.default_to(zero.item()) for tensor in tensors]
    pembeds_fv : Set[EmbeddingVar] = set()
    index_to_vembed : Dict[Any, Embedding] = {}
    subst : Subst = {}
    freshened_tensors = []
    result_is_zero = False
    for (i, (tensor, input)) in enumerate(zip(tensors, inputs)):
        if not tensor.isdisjoint(pembeds_fv): tensor = tensor.freshen()
        freshened_tensors.append(tensor)
        pembeds_fv.update(tensor.pembeds)
        for vembed, index in zip(tensor.vembeds, input):
            if index in index_to_vembed:
                if not index_to_vembed[index].unify(vembed, subst):
                    result_is_zero = True
            else:
                index_to_vembed[index] = vembed
    output_vembeds = tuple(index_to_vembed.pop(index).clone(subst) for index in output)
    # Remaining entries in index_to_vembed are what's summed out, ordered by first appearance in inputs
    if result_is_zero:
        # TODO: represent all-zero tensor with empty physical?
        outsize = tuple(e.numel() for e in output_vembeds)
        return (EmbeddedTensor(zero.expand(outsize), default=zero.item()),
                EmbeddedTensor(zero.new_zeros((), dtype=torch.long)
                                   .expand(outsize + (len(output_vembeds),)), default=0))
    projected_tensors = [project(tensor.physical, None, tensor.pembeds, subst)
                         for tensor in freshened_tensors]
    pembed_to_char = dict(zip(chain.from_iterable(pembeds for (view, pembeds) in projected_tensors),
                              map(chr, count(ord('a')))))
    output_pembeds_set = frozenset(k for e in output_vembeds for k in e.fv(subst))
    output_pembeds = tuple(output_pembeds_set)
    equation = ','.join(''.join(pembed_to_char[k] for k in pembeds)
                        for (view, pembeds) in projected_tensors) \
             + '->' + ''.join(pembed_to_char[k] for k in output_pembeds)
    #print(equation, file=stderr)
    compiled = torch_semiring_einsum.compile_equation(equation)
    out, ptr = torch_semiring_einsum.log_viterbi_einsum_forward(compiled,
                 *(view for (view, pembed) in projected_tensors), block_size=1)
                                                                  #TODO: why block_size=1?
    assert(len(output_pembeds) == out.ndim == ptr.ndim - 1)
    assert(len(pembed_to_char) == len(output_pembeds) + ptr.size(-1))
    pembed_to_ptr = dict(chain(((k, torch.arange(k._numel)
                                         .view(tuple(chain((-1,), repeat(1, out.ndim-1))))
                                         .movedim(0, i)
                                         .expand(out.size()))
                                for i, k in enumerate(output_pembeds)),
                               zip((k for k in pembed_to_char.keys()
                                      if k not in output_pembeds_set),
                                   ptr.movedim(-1,0))))
    ptrs : List[Tensor] = []
    for e in index_to_vembed.values():
        o, s = e.stride(subst)
        p = ptr.new_tensor(o, dtype=torch.long).expand(out.size())
        for k, alpha in s.items(): p = p.add(pembed_to_ptr[k], alpha=alpha)
        ptrs.append(p)
    k = EmbeddingVar(ptr.size(-1))
    assert(k.numel() == len(ptrs))
    return (EmbeddedTensor(out, output_pembeds, output_vembeds, default=zero.item()),
            EmbeddedTensor(torch.stack(ptrs), (k,) + output_pembeds, output_vembeds + (k,), default=0))
