"""
# Algebraic index types

A tensor axis often corresponds to a random variable, and indices along
the axis often correspond to possible values of the variable.  Because the
variable can be of product or sum type, the range of indices [0,n) is often
[0,l*m) representing a Cartesian product, [0,l+m) representing a disjoint
union, or a nesting of these operations.  This algebraic type structure affects
our computations (such as einsum and solve) because the input tensors
are often sparse factors that pack/unpack product/sum types, such as
eye(6).reshape(6,2,3) or eye(2,3).  We represent these sparse tensors
compactly for asymptotic speedups.

The key is the following language of "axes", which denote injective mappings
from "physical" indices to a "virtual" index:

    Axis ::= X(numel)                -- PhysicalAxis
           | Axis * ... * Axis       -- ProductAxis
           | numel + Axis + numel    -- SumAxis

For example, the sparsity pattern by which a 2*3 "physical" matrix stores a
three-axis "virtual" tensor might be described by the three axes

    [X(2) * Y(3), X(2), Y(3)].

Here X(2) and Y(3) represent an index i in [0,2) and an index j in [0,3) that
together can be used to index into the 2*3 physical matrix.  The three axes
determine the shape of the virtual tensor to be [6, 2, 3].  Element (i,j) of
the matrix corresponds to element (3*i+j,i,j) of the tensor.  Other elements of
the tensor default to zero (of some semiring).  In other words, the virtual
tensor is

    [[[p[0,0], 0     , 0     ], [0     , 0     , 0     ]],
     [[0     , p[0,1], 0     ], [0     , 0     , 0     ]],
     [[0     , 0     , p[0,2]], [0     , 0     , 0     ]],
     [[0     , 0     , 0     ], [p[1,0], 0     , 0     ]],
     [[0     , 0     , 0     ], [0     , p[1,1], 0     ]],
     [[0     , 0     , 0     ], [0     , 0     , p[1,2]]]]

where p is the 2*3 physical matrix.

To take another example, the sparsity pattern by which a length-6 "physical"
vector stores the second diagonal of a 7*6 "virtual" matrix is described by
the two axes

    [1 + Z(6) + 0, Z(6)].

Here Z(6) represents an index k in [0,6) that can be used to index into the
length-6 physical vector.  The two axes determine the shape of the virtual
tensor to be [7, 6].  Element k of the matrix corresponds to element (1+k,k) of
the matrix.  Other elements of the matrix default to zero.  In other words, the
virtual tensor is

    [[0   , 0   , 0   , 0   , 0   , 0   ],
     [q[0], 0   , 0   , 0   , 0   , 0   ],
     [0   , q[1], 0   , 0   , 0   , 0   ],
     [0   , 0   , q[2], 0   , 0   , 0   ],
     [0   , 0   , 0   , q[3], 0   , 0   ],
     [0   , 0   , 0   , 0   , q[4], 0   ],
     [0   , 0   , 0   , 0   , 0   , q[5]]]

where q is the length-6 physical vector.

So, a PhysicalAxis Z(6) represents a "physical" index in the range [0,6),
and an axis expression represents an injective mapping from the physical
indices represented by its "free" PhysicalAxes to a virtual index.  The
axes define an affine transform from the storage offset and strides of
the virtual tensor (if it ever gets materialized) to the storage offset and
strides of the physical tensor (a view on the virtual).

In sum, a virtual tensor is represented by
- a physical tensor,
- a sequence of "physical" PhysicalAxes (ordered according to the axes
  of the physical tensor), and
- a sequence of "virtual" axes (ordered according to the axes of
  the virtual tensor) containing exactly those PhysicalAxes.
We store these three pieces of information together in a PatternedTensor.
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

__all__ = ['Axis', 'PhysicalAxis', 'ProductAxis', 'SumAxis',
           'PatternedTensor', 'stack', 'einsum', 'log_viterbi_einsum_forward']

Rename = Dict["PhysicalAxis", "PhysicalAxis"]
# populated by Axis.freshen

Subst = Dict["PhysicalAxis", "Axis"]
# populated by Axis.unify

AntiSubst = Tuple[Dict[Tuple["Axis", "Axis"], "PhysicalAxis"],
                  Dict["PhysicalAxis", Tuple["Axis", "Axis"]]]
# populated by Axis.antiunify

NumberType = Union[bool, int, float] # omit the complex type to work around https://github.com/pytorch/pytorch/pull/91345

def make_letterer(alphabet: str = ascii_uppercase) -> Callable[[Any], str]:
    # A letterer is a function that turns objects into strings.  It should
    # always turn the same object into the same string, and different objects
    # into different strings.  The function produced here turns each
    # never-seen-before object into the next string in the sequence
    # A, B, C, ..., Z, AA, AB, ..., ZZ, ABC, ...
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

class Axis(ABC):
    """An injective mapping from physical indices to virtual indices."""

    @abstractmethod
    def depict(self, letterer: Callable[[PhysicalAxis], str]) -> str:
        """Produce a string representation of this axis, using the given
           callback function to turn a PhysicalAxis into a string name."""
        pass

    @abstractmethod
    def numel(self) -> int:
        """The virtual indices in the image of this mapping lie in [0,numel())."""
        pass

    @abstractmethod
    def fv(self, subst: Subst) -> Iterator[PhysicalAxis]:
        """Return the set of all PhysicalAxes in self."""
        pass

    @abstractmethod
    def stride(self, subst: Subst) -> Tuple[int, Dict[PhysicalAxis, int]]:
        """The coefficients of the affine map from physical to virtual indices.

           If (offset, strides) = self.stride(), then the physical indices
               {x1:i1, x2:i2, ...}
           map to the virtual index
               offset + strides[x1] * i1 + strides[x2] * i2 + ...."""
        pass

    def prime_factors(self, subst: Subst) -> Iterator[Axis]:
        """Express this axis as a product of non-product axes."""
        yield self

    def lookup(self, subst: Subst) -> Axis:
        """Look in subst for the end of the forwarding chain starting with self."""
        return self

    @abstractmethod
    def clone(self, subst: Subst) -> Axis:
        """Apply the given substitution, creating new Axis objects except
           reuse existing PhysicalAxis objects."""
        pass

    @abstractmethod
    def alpha(e, f: Axis, rename: Rename) -> bool:
        """Check whether the given renaming turns e into f.  Each PhysicalAxis
           in e must be a key in the given renaming, even if it's also in f."""
        pass

    @abstractmethod
    def freshen(self, rename: Rename) -> Axis:
        """Copy self while performing the given renaming, extending the
           renaming whenever necessary."""
        pass

    @abstractmethod
    def reassociate(self) -> Iterator[Axis]:
        """Remove PhysicalAxes of size 1, and merge nested ProductAxes."""
        pass

    def unify(e: Axis, f: Axis, subst: Subst) -> bool:
        """Unify the two axes by extending subst. Return success.

           Unifying two axes amounts to taking the intersection of two sparsity
           patterns.  This is useful for multiplying patterned tensors, because
           zero times anything is zero."""
        e = e.lookup(subst)
        f = f.lookup(subst)
        if e is f: return True
        if isinstance(e, ProductAxis) and isinstance(f, ProductAxis):
            if len(e.factors) == len(f.factors):
                return all(e1.unify(f1, subst) for e1, f1 in zip(e.factors, f.factors))
            else:
                warn(f"Attempt to unify {e.depict(debugging_letterer)} and {f.depict(debugging_letterer)} indicates index type mismatch")
                return False
        if isinstance(e, SumAxis) and isinstance(f, SumAxis):
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
        if isinstance(e, PhysicalAxis):
            subst[e] = f
            return True
        if isinstance(f, PhysicalAxis):
            subst[f] = e
            return True
        warn(f"Attempt to unify {e.depict(debugging_letterer)} and {f.depict(debugging_letterer)} indicates index type mismatch")
        return False

    def antiunify(e: Axis, f: Axis, antisubst: AntiSubst) -> Axis:
        """Antiunify the two axes by extending antisubst.  Return least
           general generalization (whose variables are all in antisubst).

           Antiunifying two axes amounts to finding the most precise way to
           represent the union of two sparsity patterns.  This is useful for
           adding patterned tensors, because zero plus zero is zero."""
        if __debug__:
            if e.numel() != f.numel():
                warn(f"Attempt to antiunify {e.depict(debugging_letterer)} and {f.depict(debugging_letterer)} indicates index type mismatch")
        if isinstance(e, ProductAxis) and isinstance(f, ProductAxis) and \
           len(e.factors) == len(f.factors):
            return ProductAxis(tuple(e1.antiunify(f1, antisubst)
                                     for e1, f1 in zip(e.factors, f.factors)))
        if isinstance(e, SumAxis) and isinstance(f, SumAxis) and \
           e.before == f.before and e.after == f.after:
            return SumAxis(e.before, e.term.antiunify(f.term, antisubst), e.after)
        if (e, f) in antisubst[0]:
            return antisubst[0][(e, f)]
        new = PhysicalAxis(e.numel())
        antisubst[0][(e, f)] = new
        antisubst[1][new] = (e, f)
        return new

@dataclass(eq=False, frozen=True) # The identity of PhysicalAxis objects matters,
# not just its content field _numel, because we need to distinguish the axes of
# a physical tensor even when their _numel is equal (for instance, the physical
# tensor might be a square matrix).  We also need to distinguish the axes of
# different physical tensors, which is why we cannot replace PhysicalAxis
# objects by sequential ints (0 for the first axes of a physical tensor, 1 for
# the second, etc.).
class PhysicalAxis(Axis):
    _numel: int

    def __repr__(self) -> str:
        return f"PhysicalAxis(numel={self._numel}, id={id(self)})"

    def depict(self, letterer: Callable[[PhysicalAxis], str]) -> str:
        # Produce a string like "X(2)"
        return f"{letterer(self)}({self._numel})"

    def numel(self) -> int:
        return self._numel

    def fv(self, subst: Subst) -> Iterator[PhysicalAxis]:
        look = self.lookup(subst)
        if look is self:
            yield self
        else:
            yield from look.fv(subst)

    def stride(self, subst: Subst) -> Tuple[int, Dict[PhysicalAxis, int]]:
        look = self.lookup(subst)
        return (0, {self: 1}) if look is self else look.stride(subst)

    def prime_factors(self, subst: Subst) -> Iterator[Axis]:
        look = self.lookup(subst)
        if look is self:
            yield self
        else:
            yield from look.prime_factors(subst)

    def lookup(self, subst: Subst) -> Axis:
        if self in subst:
            subst[self] = ret = subst[self].lookup(subst)
            return ret
        else:
            return self

    def clone(self, subst: Subst) -> Axis:
        e = subst.get(self, self)
        return self if e is self else e.clone(subst)

    def alpha(e, f: Axis, rename: Rename) -> bool:
        return cast(Dict[Axis, PhysicalAxis], rename).get(e, None) is f

    def freshen(self, rename: Rename) -> PhysicalAxis:
        if self in rename:
            return rename[self]
        else:
            rename[self] = ret = PhysicalAxis(self._numel)
            return ret

    def reassociate(self) -> Iterator[Axis]:
        if self._numel != 1:
            yield self

@dataclass(frozen=True)
class ProductAxis(Axis):
    factors: Tuple[Axis, ...]

    def depict(self, letterer: Callable[[PhysicalAxis], str]) -> str:
        # Produce a string like "(X(2) * Y(3))"
        return f"({'*'.join(e.depict(letterer) for e in self.factors)})"

    def numel(self) -> int:
        return reduce(mul, (e.numel() for e in self.factors), 1)

    def fv(self, subst: Subst) -> Iterator[PhysicalAxis]:
        for e in self.factors:
            yield from e.fv(subst)

    def stride(self, subst: Subst) -> Tuple[int, Dict[PhysicalAxis, int]]:
        offset: int = 0
        stride: Dict[PhysicalAxis, int] = {}
        for e in self.factors:
            if offset or stride:
                n = e.numel()
                offset *= n
                for k in stride: stride[k] *= n
            (o, s) = e.stride(subst)
            offset += o
            for k in s: stride[k] = stride.get(k, 0) + s[k]
        return (offset, stride)

    def prime_factors(self, subst: Subst) -> Iterator[Axis]:
        for e in self.factors:
            yield from e.prime_factors(subst)

    def clone(self, subst: Subst) -> Axis:
        return ProductAxis(tuple(e.clone(subst) for e in self.factors))

    def alpha(e, f: Axis, rename: Rename) -> bool:
        return isinstance(f, ProductAxis) and \
               len(e.factors) == len(f.factors) and \
               all(e1.alpha(f1, rename) for e1, f1 in zip(e.factors, f.factors))

    def freshen(self, rename: Rename) -> Axis:
        return ProductAxis(tuple(e.freshen(rename) for e in self.factors))

    def reassociate(self) -> Iterator[Axis]:
        for e in self.factors:
            yield from e.reassociate()

@dataclass(frozen=True)
class SumAxis(Axis):
    before: int
    term: Axis
    after: int

    def depict(self, letterer: Callable[[PhysicalAxis], str]) -> str:
        # Produce a string like "(1 + Z(6) + 0)"
        return f"({self.before} + {self.term.depict(letterer)} + {self.after})"

    def numel(self) -> int:
        return self.before + self.term.numel() + self.after

    def fv(self, subst: Subst) -> Iterator[PhysicalAxis]:
        yield from self.term.fv(subst)

    def stride(self, subst: Subst) -> Tuple[int, Dict[PhysicalAxis, int]]:
        (o, s) = self.term.stride(subst)
        return (o + self.before, s)

    def clone(self, subst: Subst) -> Axis:
        return SumAxis(self.before, self.term.clone(subst), self.after)

    def alpha(e, f: Axis, rename: Rename) -> bool:
        return isinstance(f, SumAxis) and \
               e.before == f.before and e.after == f.after and \
               e.term.alpha(f.term, rename)

    def freshen(self, rename: Rename) -> Axis:
        return SumAxis(self.before, self.term.freshen(rename), self.after)

    def reassociate(self) -> Iterator[Axis]:
        factors = tuple(self.term.reassociate())
        yield SumAxis(self.before,
                      factors[0] if len(factors) == 1 else ProductAxis(factors),
                      self.after)

def project(virtual: Tensor,
            paxes: Optional[Sequence[PhysicalAxis]],
            vaxes: Sequence[Axis],
            subst: Subst) -> Tuple[Tensor, Sequence[PhysicalAxis]]:
    """Extract a view of the given tensor, so that indexing into the returned
       tensor according to paxes is equivalent to indexing into the given
       tensor according to vaxes."""
    if __debug__:
        if virtual.size() != Size(e.numel() for e in vaxes):
            raise ValueError(f"project(tensor of {virtual.size()}, ..., vaxes of {Size(e.numel() for e in vaxes)}")
    if not subst and paxes is None \
                 and all(isinstance(e, PhysicalAxis) for e in vaxes) \
                 and len(frozenset(vaxes)) == virtual.ndim:
        # Try to optimize for a common case
        return (virtual, cast(Sequence[PhysicalAxis], vaxes))
    offset = virtual.storage_offset()
    stride : Dict[PhysicalAxis, int] = {}
    for e, n in zip(vaxes, virtual.stride()):
        o, s = e.stride(subst)
        offset += o * n
        for k in s: stride[k] = stride.get(k, 0) + s[k] * n
    if paxes is None:
        paxes = tuple(stride.keys())
    else:
        if __debug__:
            vaxes_fv = frozenset(stride.keys())
            paxes_fv = frozenset(paxes)
            if vaxes_fv != paxes_fv:
                raise ValueError(f"project(..., paxes with {paxes_fv}, vaxes with {vaxes_fv})")
    return (virtual.as_strided(tuple(k._numel  for k in paxes),
                               tuple(stride[k] for k in paxes),
                               offset),
            paxes)

def reshape_or_view(f: Callable[[Tensor, List[int]], Tensor],
                    self: PatternedTensor,
                    *shape: Union[int, Sequence[int]]) -> PatternedTensor:
    """Produce a new PatternedTensor, with the same elements when flattened,
       whose size() is equal to s."""
    s = list(n for arg in shape for n in ((arg,) if isinstance(arg, int) else arg))
    if self.physical.numel() <= 1:
        return PatternedTensor(f(self.physical, s), default=self.default)
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
    primes : Iterator[Axis] = (prime for e in self.vaxes
                                     for prime in e.prime_factors(subst))
    packs : List[List[Axis]] = []
    for goal in s:
        pack = []
        while goal != 1:
            try:
                prime = next(primes)
            except StopIteration:
                raise RuntimeError(f'Cannot reshape_or_view {self} to {s}')
            pn = prime.numel()
            if pn > goal:
                if isinstance(prime, PhysicalAxis) and 0 == pn % goal:
                    p1 = PhysicalAxis(goal)
                    p2 = PhysicalAxis(pn // goal)
                    subst[prime] = ProductAxis((p1, p2))
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
    paxes = tuple(cast(PhysicalAxis, k) for e in self.paxes
                                        for k in e.prime_factors(subst))
    vaxes = tuple(pack[0].clone(subst) if len(pack) == 1
                  else ProductAxis(tuple(e.clone(subst) for e in pack))
                  for pack in packs)
    assert(len(vaxes) == len(s) and
           all(e.numel() == goal for e, goal in zip(vaxes, s)))
    return PatternedTensor(f(self.physical, list(k._numel for k in paxes)),
                           paxes, vaxes, self.default).reassociate()

def broadcast(vaxess: Sequence[Sequence[Axis]]) \
    -> Tuple[Tuple[List[Axis], List[PhysicalAxis]], ...]:
    """Apply broadcast semantics to the given vaxis sequences
       to make them equal in length (maximum ndim) and size,
       creating fresh PhysicalAxes for each broadcast axis.
       We also return lists of the fresh PhysicalAxes created."""
    rets : Tuple[Tuple[List[Axis], List[PhysicalAxis]], ...] \
         = tuple(([], []) for vaxes in vaxess)
    for es in zip_longest(*(reversed(vaxes) for vaxes in vaxess),
                          fillvalue=ProductAxis(())):
        ns = tuple(e.numel() for e in es)
        Ns = frozenset(n for n in ns if n != 1)
        if len(Ns) > 1: raise RuntimeError(f"Size mismatch {ns}")
        N = tuple(Ns)[0] if Ns else 1
        for ret, e, n in zip(rets, es, ns):
            if n != N:
                k = PhysicalAxis(N)
                e = k if e == ProductAxis(()) else ProductAxis((e, k))
                ret[1].insert(0, k)
            ret[0].insert(0, e)
    return rets

@dataclass
class PatternedTensor:
    physical: Tensor
    paxes   : Sequence[PhysicalAxis] = cast(Sequence[PhysicalAxis], None) # set to non-None in __post_init__
    vaxes   : Sequence[Axis]         = cast(Sequence[Axis],         None) # set to non-None in __post_init__
    default : NumberType             = 0

    def __post_init__(self):
        # Default to the trivial (dense) axis, to convert from Tensor
        if self.paxes is None:
            self.paxes = tuple(PhysicalAxis(n) for n in self.physical.size())
        else:
            if __debug__:
                if self.physical.size() != Size(k.numel() for k in self.paxes):
                    raise ValueError(f"PatternedTensor(tensor of {self.physical.size()}, paxes of {Size(k.numel() for k in self.paxes)}, ...)")
        if self.vaxes is None:
            self.vaxes = self.paxes

    @staticmethod
    def from_int(x: int, semiring: Semiring) -> PatternedTensor:
        return PatternedTensor(semiring.from_int(x),
                               default=semiring.from_int(0).item())

    @staticmethod
    def eye(size: int, semiring: Semiring) -> PatternedTensor:
        k = PhysicalAxis(size)
        return PatternedTensor(semiring.from_int(1).expand(size), (k,), (k,k),
                               semiring.from_int(0).item())

    def depict(self, letterer: Callable[[PhysicalAxis], str]) -> str:
        """Produce a string summary of this PatternedTensor, using the given
           callback function to turn a PhysicalAxis into a string name."""
        P = (k.depict(letterer) for k in self.paxes)
        V = (e.depict(letterer) for e in self.vaxes)
        return f"[\\{' '.join(P)} -> {', '.join(V)} | default {self.default}]"

    @property
    def shape(self):
        return self.size()

    def numel(self) -> int:
        return reduce(mul, (e.numel() for e in self.vaxes), 1)

    def size(self) -> Size:
        return Size(e.numel() for e in self.vaxes)

    def dim(self) -> int:
        return len(self.vaxes)

    ndimension = dim
    ndim = property(dim)

    @property
    def dtype(self) -> torch.dtype:
        return self.physical.dtype

    @property
    def requires_grad(self) -> bool:
        return self.physical.requires_grad

    def requires_grad_(self, requires_grad: bool = True) -> PatternedTensor:
        self.physical.requires_grad_(requires_grad)
        return self

    def is_complex(self) -> bool:
        return self.physical.is_complex()

    def nonphysical(self):
        return (self.paxes, self.vaxes, self.default)

    def freshen(self) -> PatternedTensor:
        """Return a new PatternedTensor (with same underlying physical storage)
           with fresh PhysicalAxes."""
        rename : Rename = {}
        return PatternedTensor(self.physical,
                               tuple(k.freshen(rename) for k in self.paxes),
                               tuple(e.freshen(rename) for e in self.vaxes),
                               self.default)

    def reassociate(self) -> PatternedTensor:
        """Remove physical axes of size 1, and merge nested ProductAxes."""
        return PatternedTensor(self.physical.squeeze(),
                               tuple(k for k in self.paxes if k._numel != 1),
                               tuple(factors[0] if len(factors) == 1 else ProductAxis(factors)
                                     for e in self.vaxes
                                     for factors in (tuple(e.reassociate()),)),
                               self.default)

    def default_to(self, default: NumberType) -> PatternedTensor:
        return self if self.default == default or isnan(self.default) and isnan(default) \
               else PatternedTensor(self.to_dense(), default=default)

    def clone(self) -> PatternedTensor:
        rename : Rename = {}
        return PatternedTensor(self.physical.clone(),
                               tuple(k.freshen(rename) for k in self.paxes),
                               tuple(e.freshen(rename) for e in self.vaxes),
                               self.default)

    def detach(self: PatternedTensor) -> PatternedTensor:
        rename : Rename = {}
        return PatternedTensor(self.physical.detach(),
                               tuple(k.freshen(rename) for k in self.paxes),
                               tuple(e.freshen(rename) for e in self.vaxes),
                               self.default)

    def copy_(self, src: PatternedTensor) -> None:
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
        self.paxes = tuple(k.freshen(rename) for k in src.paxes)
        self.vaxes = tuple(e.freshen(rename) for e in src.vaxes)
        self.default = src.default

    def isdisjoint(self, other: Union[Set[PhysicalAxis], PatternedTensor]) -> bool:
        """Check whether the PhysicalAxes in self and other overlap."""
        return (frozenset(other.paxes) if isinstance(other, PatternedTensor) else other) \
               .isdisjoint(self.paxes)

    def to_dense(self) -> Tensor:
        """Expand a physical tensor to a mostly-zero virtual tensor."""
        if len(self.paxes) == len(self.vaxes) and \
           all(isinstance(e, PhysicalAxis) for e in self.vaxes) and \
           frozenset(self.paxes) == frozenset(self.vaxes):
            # vaxes is just a permutation of paxes, so just clone view on physical
            return project(self.physical,
                           cast(Tuple[PhysicalAxis], self.vaxes),
                           self.paxes, {})[0].clone()
        virtual = self.physical.new_full(self.size(), self.default)
        project(virtual, self.paxes, self.vaxes, {})[0].copy_(self.physical)
        return virtual

    def __float__(self) -> float:
        return float(self.physical)

    def masked_fill_into(self, dest: Tensor, value: NumberType) -> None:
        """Fill elements of dest tensor with value where self is True."""
        projected = project(dest, self.paxes, self.vaxes, {})[0]
        if self.default:
            preserved = torch.where(self.physical, value, projected)
            dest.fill_(value)
            projected.copy_(preserved)
        else:
            projected.masked_fill_(self.physical, value)

    def dim_to_dense(self, dim: int) -> PatternedTensor:
        """Expand to an equivalent PatternedTensor but make sure that the given
           axis is dense and independent of the other axes."""
        vaxes = list(self.vaxes)
        e_dense = vaxes.pop(dim)
        rename : Rename = {}
        vaxes = list(e.freshen(rename) for e in vaxes)
        if isinstance(e_dense, PhysicalAxis) and e_dense not in rename:
            return self
        fv  = tuple(rename.keys())
        fv0 = tuple(rename.values())
        k = PhysicalAxis(e_dense.numel())
        vaxes.insert(dim, k)
        paxes = fv0 + (k,)
        default = self.default
        physical = self.physical.new_full(tuple(e._numel for e in paxes), default)
        project(physical, self.paxes, fv + (e_dense,), {})[0].copy_(self.physical)
        return PatternedTensor(physical, paxes, vaxes, default)

    def any(self, dim: int, keepdim: bool = False) -> PatternedTensor:
        vaxes = list(self.vaxes)
        if keepdim:
            vaxes[dim] = ProductAxis(())
        else:
            del vaxes[dim]
        ks = frozenset(self.vaxes[dim].fv({})) - frozenset(k for e in vaxes for k in e.fv({}))
        paxes = tuple(k for k in self.paxes if k not in ks)
        if self.default and self.vaxes[dim].numel() > reduce(mul, (k._numel for k in ks), 1):
            physical = self.physical.new_ones(()).expand(tuple(k._numel for k in paxes))
        else:
            physical = self.physical
            for i in range(len(self.paxes)-1, -1, -1):
                if self.paxes[i] in ks:
                    physical = physical.any(dim=i, keepdim=False)
        return PatternedTensor(physical, paxes, vaxes, self.default)

    def permute(self, dims: Sequence[int]) -> PatternedTensor:
        assert(len(dims) == len(self.vaxes))
        assert(frozenset(dims) == frozenset(range(0,len(dims))))
        return PatternedTensor(self.physical,
                               self.paxes,
                               tuple(self.vaxes[i] for i in dims),
                               self.default)

    def __iter__(self) -> Iterator[PatternedTensor]:
        self = self.dim_to_dense(0)
        vaxes = list(self.vaxes)
        k = vaxes.pop(0)
        assert(isinstance(k, PhysicalAxis))
        paxes = list(self.paxes)
        i = paxes.index(k)
        paxes.pop(i)
        perm = (i, *range(0, i), *range(i+1, len(self.paxes)))
        for p in self.physical.permute(perm):
            yield PatternedTensor(p, paxes, vaxes, self.default)

    def neg_(self) -> PatternedTensor:
        self.default = -self.default
        self.physical.neg_()
        return self

    def log_(self) -> PatternedTensor:
        self.default = log(self.default) if self.default else -inf
        self.physical.log_()
        return self

    def log1p_(self) -> PatternedTensor:
        self.default = log1p(self.default)
        self.physical.log1p_()
        return self

    def relu_(self) -> PatternedTensor:
        self.default = max(0, self.default)
        self.physical.relu_()
        return self

    def abs_(self) -> PatternedTensor:
        self.default = abs(self.default)
        self.physical.abs_()
        return self

    def nan_to_num_(self, nan: float = 0.,
                          posinf: Optional[float] = None,
                          neginf: Optional[float] = None) -> PatternedTensor:
        if isnan(self.default):
            self.default = nan
        elif isinf(self.default):
            if self.default >= 0:
                self.default = float_info.max if posinf is None else posinf
            else:
                self.default = -float_info.max if neginf is None else neginf
        self.physical.nan_to_num_(nan=nan, posinf=posinf)
        return self

    def __imul__(self, other: NumberType) -> PatternedTensor:
        self.default  *= other
        self.physical *= other
        return self

    def __itruediv__(self, other: NumberType) -> PatternedTensor:
        self.default  /= other
        self.physical /= other
        return self

    def lt(self, other: Union[float, PatternedTensor]) -> PatternedTensor:
        if isinstance(other, PatternedTensor):
            return self.binary(other, self.default < other.default, torch.lt)
        else:
            return PatternedTensor(self.physical.lt(other), self.paxes, self.vaxes, self.default < other)

    def le(self, other: Union[float, PatternedTensor]) -> PatternedTensor:
        if isinstance(other, PatternedTensor):
            return self.binary(other, self.default <= other.default, torch.le)
        else:
            return PatternedTensor(self.physical.le(other), self.paxes, self.vaxes, self.default <= other)

    def gt(self, other: Union[float, PatternedTensor]) -> PatternedTensor:
        if isinstance(other, PatternedTensor):
            return self.binary(other, self.default > other.default, torch.gt)
        else:
            return PatternedTensor(self.physical.gt(other), self.paxes, self.vaxes, self.default > other)

    def ge(self, other: Union[float, PatternedTensor]) -> PatternedTensor:
        if isinstance(other, PatternedTensor):
            return self.binary(other, self.default >= other.default, torch.ge)
        else:
            return PatternedTensor(self.physical.ge(other), self.paxes, self.vaxes, self.default >= other)

    def eq(self, other: Union[float, PatternedTensor]) -> PatternedTensor:
        if isinstance(other, PatternedTensor):
            return self.binary(other, self.default == other.default, torch.eq)
        else:
            return PatternedTensor(self.physical.eq(other), self.paxes, self.vaxes, self.default == other)

    def to(self, dtype: torch.dtype) -> PatternedTensor:
        return PatternedTensor(self.physical.to(dtype=dtype), self.paxes, self.vaxes,
                               self.physical.new_tensor(self.default, dtype=dtype).item())

    def abs(self) -> PatternedTensor:
        return PatternedTensor(self.physical.abs(), self.paxes, self.vaxes, abs(self.default))

    def exp(self) -> PatternedTensor:
        return PatternedTensor(self.physical.exp(), self.paxes, self.vaxes, exp(self.default))

    def expm1(self) -> PatternedTensor:
        return PatternedTensor(self.physical.expm1(), self.paxes, self.vaxes, expm1(self.default))

    def log(self) -> PatternedTensor:
        return PatternedTensor(self.physical.log(), self.paxes, self.vaxes,
                               log(self.default) if self.default else -inf)

    def logical_not(self) -> PatternedTensor:
        return PatternedTensor(~self.physical, self.paxes, self.vaxes, not self.default)

    def __mul__(self, other: NumberType) -> PatternedTensor:
        return PatternedTensor(self.physical * other, self.paxes, self.vaxes, self.default * other)

    def __truediv__(self, other: NumberType) -> PatternedTensor:
        return PatternedTensor(self.physical / other, self.paxes, self.vaxes, self.default / other)

    def log_softmax(self, dim) -> PatternedTensor:
        if dim < 0: dim += self.ndim
        self = self.dim_to_dense(dim)
        k = self.vaxes[dim]
        i = self.paxes.index(k)
        return PatternedTensor(self.physical.log_softmax(dim = i),
                               self.paxes, self.vaxes, -log(k._numel))

    def equal_default(self) -> bool:
        return cast(bool, self.physical.eq(self.default).all().item())

    def allclose_default(self, rtol=1e-05, atol=1e-08) -> bool:
        return self.physical.allclose(self.physical.new_tensor(self.default),
                                      rtol=rtol, atol=atol, equal_nan=True)

    def equal(self, other: PatternedTensor) -> bool:
        s = self.size()
        if s != other.size(): return False
        n = s.numel()
        if not self.isdisjoint(other): other = other.freshen()
        selfok  = self.physical == other.default
        otherok = other.physical == self.default
        # Compare overlapping parts to each other
        subst : Subst = {}
        if all(v.unify(u, subst) for v, u in zip(self.vaxes, other.vaxes)):
            # There is overlap
            subself, subaxes = project(self.physical, None, self.paxes, subst)
            subother, _ = project(other.physical, subaxes, other.paxes, subst)
            if not subself.equal(subother): return False
            project(selfok , None, self .paxes, subst)[0].fill_(True)
            project(otherok, None, other.paxes, subst)[0].fill_(True)
            n += subself.numel()
        # Compare defaults and non-overlapping parts
        return (n <= selfok.numel() + otherok.numel() or
                self.default == other.default) and \
               bool(selfok.all()) and bool(otherok.all())

    def allclose(self, other: PatternedTensor, rtol=1e-05, atol=1e-08, equal_nan=False) -> bool:
        s = self.size()
        if s != other.size(): return False
        n = s.numel()
        if not self.isdisjoint(other): other = other.freshen()
        selfok = self.physical.isclose(other.physical.new_tensor(other.default), rtol=rtol, atol=atol, equal_nan=equal_nan)
        otherok = self.physical.new_tensor(self.default).isclose(other.physical, rtol=rtol, atol=atol, equal_nan=equal_nan)
        # Compare overlapping parts to each other
        subst : Subst = {}
        if all(v.unify(u, subst) for v, u in zip(self.vaxes, other.vaxes)):
            # There is overlap
            subself, subaxes = project(self.physical, None, self.paxes, subst)
            subother, _ = project(other.physical, subaxes, other.paxes, subst)
            if not subself.allclose(subother, rtol=rtol, atol=atol, equal_nan=equal_nan): return False
            project(selfok , None, self .paxes, subst)[0].fill_(True)
            project(otherok, None, other.paxes, subst)[0].fill_(True)
            n += subself.numel()
        # Compare defaults and non-overlapping parts
        return (n <= selfok.numel() + otherok.numel() or
                self.physical.new_tensor(self.default)
                    .allclose(other.physical.new_tensor(other.default),
                              rtol=rtol, atol=atol, equal_nan=equal_nan)) and \
               bool(selfok.all()) and bool(otherok.all())

    def expansion(t, u: PatternedTensor) -> Tuple[Sequence[PhysicalAxis],
                                                  Sequence[Axis],
                                                  Sequence[PhysicalAxis],
                                                  Sequence[Axis],
                                                  Sequence[PhysicalAxis],
                                                  Sequence[Axis]]:
        """Compute how to expand two PatternedTensors in order to match."""
        antisubst : AntiSubst = ({}, {})
        paxes1 : List[PhysicalAxis] = list(t.paxes)
        paxes2 : List[PhysicalAxis] = list(u.paxes)
        lggs : List[Axis] = []
        for e, f in zip_longest(reversed(t.vaxes),
                                reversed(u.vaxes),
                                fillvalue=ProductAxis(())):
            if e == ProductAxis(()) != f:
                new = PhysicalAxis(f.numel())
                paxes1.insert(0, new)
                antisubst[0][(new, f)] = new
                antisubst[1][new] = (new, f)
                lggs.insert(0, new)
            elif f == ProductAxis(()) != e:
                new = PhysicalAxis(e.numel())
                paxes2.insert(0, new)
                antisubst[0][(e, new)] = new
                antisubst[1][new] = (e, new)
                lggs.insert(0, new)
            else:
                lggs.insert(0, e.antiunify(f, antisubst))
        (gs, es, fs) = zip(*((g, e, f) for (g, (e, f)) in antisubst[1].items())) \
                       if antisubst[1] else ((), (), ())
        return (gs, lggs, paxes1, es, paxes2, fs)

    def commutative(t, u: PatternedTensor, identity: NumberType, default: NumberType,
                    operate_: Callable[[Tensor, Tensor], Any]) -> PatternedTensor:
        """Apply a symmetric binary operation with identity."""
        (gs, lggs, paxes1, es, paxes2, fs) = t.expansion(u)
        tp = t.physical.expand(tuple(k._numel for k in paxes1))
        up = u.physical.expand(tuple(k._numel for k in paxes2))
        if t.default != identity or \
           len(paxes1) != len(t.paxes) or \
           len(paxes2) == len(u.paxes) and t.physical.numel() >= u.physical.numel():
            td = PatternedTensor(tp, paxes1, es, t.default).to_dense()
            if u.default == identity:
                tp = project(td, paxes2, fs, {})[0]
                operate_(tp, up)
            else:
                ud = PatternedTensor(up, paxes2, fs, u.default).to_dense()
                operate_(td, ud)
            return PatternedTensor(td, gs, lggs, default)
        else:
            ud = PatternedTensor(up, paxes2, fs, u.default).to_dense()
            up = project(ud, paxes1, es, {})[0]
            operate_(up, tp)
            return PatternedTensor(ud, gs, lggs, default)

    def binary(t, u: PatternedTensor, default: NumberType,
               operate: Callable[[Tensor, Tensor], Tensor]) -> PatternedTensor:
        """Apply a binary operation."""
        (gs, lggs, paxes1, es, paxes2, fs) = t.expansion(u)
        tp = t.physical.expand(tuple(k._numel for k in paxes1))
        up = u.physical.expand(tuple(k._numel for k in paxes2))
        td = PatternedTensor(tp, paxes1, es, t.default).to_dense()
        ud = PatternedTensor(up, paxes2, fs, u.default).to_dense()
        return PatternedTensor(operate(td, ud), gs, lggs, default)

    def add(t, u: PatternedTensor) -> PatternedTensor:
        return t.commutative(u, 0, t.default + u.default,
                             lambda x, y: x.add_(y))

    def mul(t, u: PatternedTensor) -> PatternedTensor:
        # TODO: if one or both of the defaults is 0, then that input's sparsity
        # pattern bounds the result's sparsity pattern (NaN be damned)
        return t.commutative(u, 1, t.default * u.default,
                             lambda x, y: x.mul_(y))

    def logaddexp(t, u: PatternedTensor) -> PatternedTensor:
        return t.commutative(u, -inf, t.physical.new_tensor(t.default).logaddexp(
                                      u.physical.new_tensor(u.default)).item(),
                             lambda x, y: torch.logaddexp(x, y, out=x))

    def maximum(t, u: PatternedTensor) -> PatternedTensor:
        return t.commutative(u, -inf, max(t.default, u.default),
                             lambda x, y: torch.maximum(x, y, out=x))

    def logical_or(t, u: PatternedTensor) -> PatternedTensor:
        # TODO: short circuiting
        return t.commutative(u, False, t.default or u.default,
                             lambda x, y: x.logical_or_(y))

    def logical_and(t, u: PatternedTensor) -> PatternedTensor:
        # TODO: short circuiting
        return t.commutative(u, True, t.default and u.default,
                             lambda x, y: x.logical_and_(y))

    def sub(t, u: PatternedTensor) -> PatternedTensor:
        """Subtract two PatternedTensors."""
        (gs, lggs, paxes1, es, paxes2, fs) = t.expansion(u)
        tp = t.physical.expand(tuple(k._numel for k in paxes1))
        default = t.default - u.default
        if t.default != 0 or \
           len(paxes1) != len(t.paxes) or \
           len(paxes2) == len(u.paxes) and t.physical.numel() >= u.physical.numel():
            up = u.physical.expand(tuple(k._numel for k in paxes2))
            td = PatternedTensor(tp, paxes1, es, t.default).to_dense()
            if u.default == 0:
                tp = project(td, paxes2, fs, {})[0]
                tp.sub_(up)
            else:
                ud = PatternedTensor(up, paxes2, fs, u.default).to_dense()
                td.sub_(ud)
            return PatternedTensor(td, gs, lggs, default)
        else:
            up = (-u.physical).expand(tuple(k._numel for k in paxes2))
            ud = PatternedTensor(up, paxes2, fs, -u.default).to_dense()
            up = project(ud, paxes1, es, {})[0]
            up.add_(tp)
            return PatternedTensor(ud, gs, lggs, default)

    __add__ = add
    __sub__ = sub

    def where(t, c, u) -> PatternedTensor:
        if c.default: t, u = u, t
        if not t.isdisjoint(c): t = t.freshen()
        # Broadcast
        (t_vaxes, _), (c_vaxes, c_new_paxes), (u_vaxes, u_new_paxes) \
            = broadcast((t.vaxes, c.vaxes, u.vaxes))
        c_paxes = tuple(chain(c_new_paxes, c.paxes))
        u_paxes = tuple(chain(u_new_paxes, u.paxes))

        # Extract the intersection of the non-default regions of t and c,
        # and match up their axes
        subst : Subst = {}
        success = all(ec.unify(et, subst) for ec, et in zip(c_vaxes, t_vaxes))
        if success:
            kst    : FrozenSet[PhysicalAxis] = frozenset(k for e in t.paxes     for k in e.fv(subst))
            ksc    : FrozenSet[PhysicalAxis] = frozenset(k for e in c.paxes     for k in e.fv(subst))
            ksc_new: FrozenSet[PhysicalAxis] = frozenset(k for e in c_new_paxes for k in e.fv(subst))
            ks     : Tuple[PhysicalAxis,...] = tuple(ksc | ksc_new)
            projected_t = project(t.physical, tuple(k for k in ks if k in kst), t.paxes, subst)[0]
            for i, k in enumerate(ks):
                if k not in kst: projected_t = projected_t.unsqueeze(i)
            projected_c = project(c.physical, tuple(k for k in ks if k in ksc), c.paxes, subst)[0]
            for i, k in enumerate(ks):
                if k not in ksc: projected_c = projected_c.unsqueeze(i)
            assert(projected_t.ndim == projected_c.ndim and
                   all(len({n1, n2} - {1}) <= 1 for n1, n2 in zip(projected_t.size(),
                                                                  projected_c.size())))

            # Is the non-default region of t a superset of the non-default region of c?
            # In other words, did unification do anything to c_paxes?
            fullness_test = frozenset(e.lookup(subst) for e in c_paxes)
            full = len(fullness_test) == len(c_paxes) \
                   and all(isinstance(e, PhysicalAxis) for e in fullness_test)

        # Expand u to c
        antisubst : AntiSubst = ({}, {})
        lggs = [ec.antiunify(eu, antisubst)
                for ec, eu in zip(c_vaxes, u_vaxes)]
        (gs, ecs, eus) = zip(*((g, ec, eu) for (g, (ec, eu)) in antisubst[1].items())) \
                         if antisubst[1] else ((), (), ())
        up = u.physical.expand(tuple(k._numel for k in u_paxes))
        ud = PatternedTensor(up, u_paxes, eus, u.default).to_dense()

        if not (success and full):
            # Fill expanded u with t.default
            up = project(ud, c_paxes, ecs, {})[0]
            up.masked_fill_(~c.physical if c.default else c.physical, t.default)
        if success:
            # Mutate expanded u with matched t
            up = project(ud, ks, ecs, subst)[0]
            up.copy_(up.where(projected_c, projected_t) if c.default
                     else projected_t.where(projected_c, up))

        return PatternedTensor(ud, gs, lggs, u.default)

    def transpose(self, dim0, dim1) -> PatternedTensor:
        if dim0 == dim1:
            return self
        else:
            if dim0 > dim1: dim0, dim1 = dim1, dim0
            vaxes = self.vaxes
            assert(0 <= dim0 < dim1 < len(vaxes))
            vaxes = (vaxes[      :dim0  ] + # type: ignore
                     vaxes[dim1  :dim1+1] +
                     vaxes[dim0+1:dim1  ] +
                     vaxes[dim0  :dim0+1] +
                     vaxes[dim1+1:      ])
            return PatternedTensor(self.physical, self.paxes, vaxes, self.default)

    def t(self) -> PatternedTensor:
        assert(len(self.vaxes) <= 2)
        if len(self.vaxes) < 2: return self
        return self.transpose(0,1)

    @property
    def T(self) -> PatternedTensor:
        return PatternedTensor(self.physical, self.paxes, self.vaxes[::-1], self.default)

    def flatten(self) -> PatternedTensor:
        if len(self.vaxes) == 1:
            return self.reassociate()
        else:
            return PatternedTensor(self.physical,
                                   self.paxes,
                                   (ProductAxis(tuple(self.vaxes)),),
                                   self.default).reassociate()

    def unsqueeze(self, dim: int) -> PatternedTensor:
        if dim < 0: dim += self.ndim + 1
        vaxes = list(self.vaxes)
        vaxes.insert(dim, ProductAxis(()))
        return PatternedTensor(self.physical, self.paxes, vaxes, self.default)

    def reshape(self, *shape: Union[int, Sequence[int]]) -> PatternedTensor:
        return reshape_or_view(lambda tensor, size: tensor.reshape(size), self, *shape)

    def view(self, *shape: Union[int, Sequence[int]]) -> PatternedTensor:
        return reshape_or_view(lambda tensor, size: tensor.view(size), self, *shape)

    def expand(self: PatternedTensor, *sizes: int) -> PatternedTensor:
        paxes: List[PhysicalAxis] = list(self.paxes)
        vaxes: List[Axis        ] = []
        for e, n in zip_longest(reversed(self.vaxes), reversed(sizes)):
            if n is None:
                raise RuntimeError(f"PatternedTensor.expand: the number of sizes provided ({len(sizes)}) must be greater or equal to the number of dimensions in the tensor ({len(self.vaxes)})")
            if e is None or e.numel() == 1 != n:
                k = PhysicalAxis(n)
                e = k if e is None or e == ProductAxis(()) else ProductAxis((e, k))
                paxes.insert(0, k)
            if e.numel() != n:
                raise RuntimeError(f"PatternedTensor.expand: cannot extend {self.size()} to {sizes}")
            vaxes.insert(0, e)
        return PatternedTensor(self.physical.expand(tuple(k._numel for k in paxes)),
                               paxes, vaxes, self.default)

    def repeat(self: PatternedTensor, *sizes: int) -> PatternedTensor:
        return self.expand(*sizes).clone()

    def solve(a, b: PatternedTensor, semiring: Semiring) -> PatternedTensor:
        """Solve x = a @ x + b for x."""
        assert(len(a.vaxes) == 2 and len(b.vaxes) >= 1)
        assert(a.vaxes[0].numel() == a.vaxes[1].numel() == b.vaxes[0].numel())
        zero = semiring.from_int(0)
        a = a.default_to(zero.item())
        b = b.default_to(zero.item())
        if not a.isdisjoint(b): b = b.freshen()
        e = b.vaxes[0] # Axis for b + a*b + ... + a^n*b (initially n=0)
                       # Invariant: a and e are disjoint
        while True: # Compute least-dense axis for solution. TODO any easier way?
            # Update e := a*e + b
            subst : Subst = {}
            if e.unify(a.vaxes[1], subst):
                antisubst : AntiSubst = ({}, {})
                e = e.antiunify(a.vaxes[0].clone(subst), antisubst)
                if all(isinstance(e1, PhysicalAxis) for e1, _ in antisubst[0]):
                    break # acquiescence
            else:
                break # a*e = 0
        # Copy e  (for naming the columns of a and rows of b)
        #   to e0 (for naming the rows of a)
        rename : Rename = {}
        e0  = e.freshen(rename)
        fv  = tuple(rename.keys())
        fv0 = tuple(rename.values())
        # Copy ProductAxis(b.vaxes[1:])
        #   to e1 (for naming the columns of b)
        rename = {}
        ebs0 = tuple(eb.freshen(rename) for eb in b.vaxes[1:])
        fvb  = tuple(rename.keys())
        fvb0 = tuple(rename.values())
        # Convert b to regular matrix tensor
        subst = {}
        if not e.unify(b.vaxes[0], subst): raise AssertionError
        projected_b = project(b.physical, None, b.paxes, subst)
        dense_b = PatternedTensor(projected_b[0],
                                  projected_b[1],
                                  (ProductAxis(fv ).clone(subst),
                                   ProductAxis(fvb).clone(subst)),
                                  b.default).to_dense()
        # Convert relevant portion of a to regular matrix tensor
        subst = {}
        if not (e0.unify(a.vaxes[0], subst) and
                e .unify(a.vaxes[1], subst)): raise AssertionError
        projected_a = project(a.physical, None, a.paxes, subst)
        dense_a = PatternedTensor(projected_a[0],
                                  projected_a[1],
                                  (ProductAxis(fv0).clone(subst),
                                   ProductAxis(fv ).clone(subst)),
                                  a.default).to_dense()
        # Solve
        x = semiring.solve(dense_a, dense_b)
        return PatternedTensor(x.reshape(tuple(k._numel for k in fv + fvb0)),
                               fv + fvb0, (e,) + ebs0, b.default)

    def mv(self, v: PatternedTensor, semiring: Semiring) -> PatternedTensor:
        return einsum((self,v), ("ij", "j"), "i", semiring)

    def mm(self, m: PatternedTensor, semiring: Semiring) -> PatternedTensor:
        return einsum((self,m), ("ij", "jk"), "ik", semiring)

def stack(tensors: Sequence[PatternedTensor], dim: int = 0) -> PatternedTensor:
    """Concatenate PatternedTensors along a new axis.
       The inputs must have the same (virtual) size and default."""
    assert(tensors)
    head, *tail = tensors
    size = head.size()
    default = head.default
    lggs = head.vaxes
    if not tail:
        lggs = list(lggs)
        lggs.insert(dim, ProductAxis(()))
        return PatternedTensor(head.physical, head.paxes, lggs, default)
    for t in tail: # Antiunify all input vaxes
        assert(size == t.size())
        assert(default == t.default)
        antisubst : AntiSubst = ({}, {})
        lggs = tuple(e.antiunify(f, antisubst)
                     for e, f in zip(lggs, t.vaxes))
    k = PhysicalAxis(len(tensors))
    ks = tuple(antisubst[1])
    paxes = list(ks)
    paxes.insert(0, k)
    vaxes = list(lggs)
    vaxes.insert(dim, k)
    physical = head.physical.new_full(tuple(e._numel for e in paxes), default)
    for t, p in zip(tensors, physical):
        subst : Subst = {}
        if not all(e.unify(f, subst)
                   for e, f in zip(lggs, t.vaxes)):
            raise AssertionError
            # Due to antiunification above, t.vaxes should be an instance of lggs.
            # So, subst should map variables in t.vaxes to only variables.
        project(p, tuple(cast(PhysicalAxis, e.lookup(subst)) for e in t.paxes),
                ks, subst)[0].copy_(t.physical)
    return PatternedTensor(physical, paxes, vaxes, default)

def depict_einsum(fun: str,
                  tensors: Sequence[PatternedTensor],
                  inputs: Sequence[Sequence[Any]],
                  output: Sequence[Any]) -> str:
    letterer = make_letterer()
    return "\n".join([fun + "("] +
                     [f"    {' '.join(map(letterer, input))} {tensor.depict(letterer)}"
                      for tensor, input in zip(tensors, inputs)] +
                     [f"    -> {' '.join(map(letterer, output))})"])

def einsum(tensors: Sequence[PatternedTensor],
           inputs: Sequence[Sequence[Any]],
           output: Sequence[Any],
           semiring: Semiring) -> PatternedTensor:
    """
    To perform an einsum operation on PatternedTensors, we start with
    PatternedTensors whose sets of physical PhysicalAxes do not overlap, but
    then we *unify* the virtual axes that get co-indexed.  For example,
    if the two example PatternedTensors above were the inputs (bcd,ab->...),
    then we would unify X(2) * Y(3) with Z(6).  Hence, before we pass the
    einsum job to torch_semiring_einsum, we need to reshape (more generally,
    view) the second physical tensor -- a length-6 vector indexed by Z -- as a
    2*3 matrix indexed by X and Y.  If unification fails, then our einsum
    returns zero; the only possible unification failure in an einsum operation
    that respects the algebraic type structure of the indices should be between
    inl and inr (1 + Z(6) + 0 fails to unify with 0 + W(1) + 6).
    """
    assert(len(tensors) == len(inputs))
    assert(len(tensor.vaxes) == len(input)
           for tensor, input in zip(tensors, inputs))
    assert(frozenset(index for input in inputs for index in input) >= frozenset(output))
    zero = semiring.from_int(0)
    one  = semiring.from_int(1)
    if len(tensors) == 0:
        return PatternedTensor(one, default=zero.item())
    #print(depict_einsum('einsum', tensors, inputs, output), file=stderr)
    tensors = [tensor.default_to(zero.item()) for tensor in tensors]
    paxes_fv : Set[PhysicalAxis] = set()
    index_to_vaxis : Dict[Any, Axis] = {}
    subst : Subst = {}
    freshened_tensors = []
    result_is_zero = False
    for (i, (tensor, input)) in enumerate(zip(tensors, inputs)):
        if not tensor.isdisjoint(paxes_fv): tensor = tensor.freshen()
        freshened_tensors.append(tensor)
        paxes_fv.update(tensor.paxes)
        for vaxis, index in zip(tensor.vaxes, input):
            if index in index_to_vaxis:
                if not index_to_vaxis[index].unify(vaxis, subst):
                    result_is_zero = True
            else:
                index_to_vaxis[index] = vaxis
    output_vaxes = tuple(index_to_vaxis[index].clone(subst) for index in output)
    if result_is_zero:
        # TODO: represent all-zero tensor with empty physical?
        outsize = tuple(e.numel() for e in output_vaxes)
        return PatternedTensor(zero.expand(outsize), default=zero.item())
    projected_tensors = [project(tensor.physical, None, tensor.paxes, subst)
                         for tensor in freshened_tensors]
    paxis_to_char = dict(zip(chain.from_iterable(paxes for view, paxes in projected_tensors),
                             map(chr, count(ord('a')))))
    output_paxes = tuple(frozenset(k for e in output_vaxes for k in e.fv(subst)))
    equation = ','.join(''.join(paxis_to_char[k] for k in paxes)
                        for view, paxes in projected_tensors) \
             + '->' + ''.join(paxis_to_char[k] for k in output_paxes)
    #print(equation, file=stderr)
    compiled = torch_semiring_einsum.compile_equation(equation)
    out = semiring.einsum(compiled, *(view for view, paxes in projected_tensors))
    assert(out.dtype == semiring.dtype)
    return PatternedTensor(out, output_paxes, output_vaxes, default=zero.item())

def log_viterbi_einsum_forward(tensors: Sequence[PatternedTensor],
                               inputs: Sequence[Sequence[Any]],
                               output: Sequence[Any],
                               semiring: Semiring) -> Tuple[PatternedTensor, PatternedTensor]:
    assert(len(tensors) == len(inputs))
    assert(len(tensor.vaxes) == len(input)
           for tensor, input in zip(tensors, inputs))
    assert(frozenset(index for input in inputs for index in input) >= frozenset(output))
    zero = semiring.from_int(0)
    one  = semiring.from_int(1)
    if len(tensors) == 0:
        return (PatternedTensor(one, default=zero.item()),
                PatternedTensor(one.new_empty((0,), dtype=torch.long), default=0))
    #print(depict_einsum('log_viterbi_einsum_forward', tensors, inputs, output), file=stderr)
    tensors = [tensor.default_to(zero.item()) for tensor in tensors]
    paxes_fv : Set[PhysicalAxis] = set()
    index_to_vaxis : Dict[Any, Axis] = {}
    subst : Subst = {}
    freshened_tensors = []
    result_is_zero = False
    for (i, (tensor, input)) in enumerate(zip(tensors, inputs)):
        if not tensor.isdisjoint(paxes_fv): tensor = tensor.freshen()
        freshened_tensors.append(tensor)
        paxes_fv.update(tensor.paxes)
        for vaxis, index in zip(tensor.vaxes, input):
            if index in index_to_vaxis:
                if not index_to_vaxis[index].unify(vaxis, subst):
                    result_is_zero = True
            else:
                index_to_vaxis[index] = vaxis
    output_vaxes = tuple(index_to_vaxis.pop(index).clone(subst) for index in output)
    # Remaining entries in index_to_vaxis are what's summed out, ordered by first appearance in inputs
    if result_is_zero:
        # TODO: represent all-zero tensor with empty physical?
        outsize = tuple(e.numel() for e in output_vaxes)
        return (PatternedTensor(zero.expand(outsize), default=zero.item()),
                PatternedTensor(zero.new_zeros((), dtype=torch.long)
                                    .expand(outsize + (len(output_vaxes),)), default=0))
    projected_tensors = [project(tensor.physical, None, tensor.paxes, subst)
                         for tensor in freshened_tensors]
    paxis_to_char = dict(zip(chain.from_iterable(paxes for (view, paxes) in projected_tensors),
                             map(chr, count(ord('a')))))
    output_paxes_set = frozenset(k for e in output_vaxes for k in e.fv(subst))
    output_paxes = tuple(output_paxes_set)
    equation = ','.join(''.join(paxis_to_char[k] for k in paxes)
                        for view, paxes in projected_tensors) \
             + '->' + ''.join(paxis_to_char[k] for k in output_paxes)
    #print(equation, file=stderr)
    compiled = torch_semiring_einsum.compile_equation(equation)
    out, ptr = torch_semiring_einsum.log_viterbi_einsum_forward(compiled,
                 *(view for view, paxes in projected_tensors))
    assert(len(output_paxes) == out.ndim == ptr.ndim - 1)
    assert(len(paxis_to_char) == len(output_paxes) + ptr.size(-1))
    paxis_to_ptr = dict(chain(((k, torch.arange(k._numel)
                                        .view(tuple(chain((-1,), repeat(1, out.ndim-1))))
                                        .movedim(0, i)
                                        .expand(out.size()))
                               for i, k in enumerate(output_paxes)),
                              zip((k for k in paxis_to_char.keys()
                                     if k not in output_paxes_set),
                                  ptr.movedim(-1,0))))
    ptrs : List[Tensor] = []
    for e in index_to_vaxis.values():
        o, s = e.stride(subst)
        p = ptr.new_tensor(o, dtype=torch.long).expand(out.size())
        for k, alpha in s.items(): p = p.add(paxis_to_ptr[k], alpha=alpha)
        ptrs.append(p)
    k = PhysicalAxis(ptr.size(-1))
    assert(k.numel() == len(ptrs))
    return (PatternedTensor(out, output_paxes, output_vaxes, default=zero.item()),
            PatternedTensor(torch.stack(ptrs), (k,) + output_paxes, output_vaxes + (k,), default=0))
