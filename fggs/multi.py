__all__ = ['MultiTensor', 'MultiShape', 'multi_mv', 'multi_solve']
from typing import Union, Tuple, TypeVar, Iterator, Dict, Mapping, MutableMapping, Optional
from fggs.semirings import Semiring
from fggs.indices import EmbeddedTensor
import torch
from torch import Tensor, Size

T = TypeVar('T')
MultiTensorKey = Union[T, Tuple[T,...]]
MultiShape = Mapping[MultiTensorKey, Size]

class MultiTensor(MutableMapping[MultiTensorKey, EmbeddedTensor]):
    """A mapping from keys to EmbeddedTensors that supports some vector and
       matrix operations. The keys can be thought of as partitioning the set
       of indices into a disjoint union whose order doesn't matter."""
    def __init__(self, shapes: Union[MultiShape, Tuple[MultiShape,...]], semiring: Semiring):
        if not isinstance(shapes, tuple):
            shapes = (shapes,)
        self.shapes = shapes
        self.semiring = semiring
        self._dict: Dict[MultiTensorKey, EmbeddedTensor] = {}

    def __getitem__(self, key: MultiTensorKey) -> EmbeddedTensor:
        try:
            return self._dict[key]
        except KeyError:
            if isinstance(key, tuple):
                shape = sum([s[x] for s, x in zip(self.shapes, key)], ())
            else:
                shape = self.shapes[0][key]
            return EmbeddedTensor.from_int(0, self.semiring).expand(*shape)
    def __setitem__(self, key: MultiTensorKey, val: EmbeddedTensor):
        if isinstance(key, tuple):
            shape = sum([s[x] for s, x in zip(self.shapes, key)], ())
        else:
            shape = self.shapes[0][key]
        if val.shape != shape:
            raise ValueError(f'expected Tensor with shape {shape}, got {val.shape}')
        self._dict[key] = val
    def __delitem__(self, key: MultiTensorKey):              del self._dict[key]
    def __contains__(self, key: MultiTensorKey):             return key in self._dict
    def __iter__(self) -> Iterator[MultiTensorKey]:          return iter(self._dict)
    def __len__(self) -> int:                                return len(self._dict)
    def __str__(self) -> str:                                return str(self._dict)
    def __repr__(self) -> str:                               return repr(self._dict)

    def allclose(self, other: 'MultiTensor', tol: float) -> bool:
        """Returns true if all elements of self and other are within tol of each other."""
        if tol == 0:
            for k, t in self.items():
                if k in other:
                    if not t.equal(other[k]): return False
                else:
                    assert(t.default == self.semiring.from_int(0).item())
                    if not t.equal_default(): return False
            for k, t in other.items():
                if k not in self:
                    assert(t.default == self.semiring.from_int(0).item())
                    if not t.equal_default(): return False
        else:
            for k, t in self.items():
                if k in other:
                    if not t.allclose(other[k], atol=tol, rtol=0.): return False
                else:
                    assert(t.default == self.semiring.from_int(0).item())
                    if not t.allclose_default(atol=tol, rtol=0.): return False
            for k, t in other.items():
                if k not in self:
                    assert(t.default == self.semiring.from_int(0).item())
                    if not t.allclose_default(atol=tol, rtol=0.): return False
        return True

    def copy_(self, other: 'MultiTensor'):
        """Copy all elements from other to self."""
        for k in self:
            if k not in other:
                del self[k]
        for k in other:
            if k in self:
                self[k].copy_(other[k])
            else:
                self[k] = other[k].clone()

    def clone(self) -> 'MultiTensor':
        c = MultiTensor(self.shapes, self.semiring)
        c.copy_(self)
        return c

    def add_single(self, k: MultiTensorKey, v: EmbeddedTensor):
        """Add v to self[k]. If self[k] does not exist, it is initialized to zero."""
        assert(v.physical.dtype == self.semiring.dtype)
        if k in self:
            self[k] = self.semiring.add(self[k], v)
        else:
            self[k] = v

    def __add__(self, other: 'MultiTensor') -> 'MultiTensor':
        result = self.clone()
        for x, t in other.items():
            result.add_single(x, t)
        return result
    
    def __sub__(self, other: 'MultiTensor') -> 'MultiTensor':
        result = self.clone()
        for x, t in other.items():
            result[x] = self.semiring.sub(result[x], t)
        return result

def multi_mv(a: MultiTensor, b: MultiTensor, transpose: bool = False) -> MultiTensor:
    """Compute the product a @ b, where the elements of a are flattened into
    matrices and the elements of b are flattened into vectors.

    Arguments:
    - transpose: compute a.T @ b = b @ a instead.
    """
    ishapes, jshapes = a.shapes
    # assert b.shapes == (jshapes,)
    flat_ishapes = {x:Size((ishapes[x].numel(),)) for x in ishapes}
    flat_jshapes = {x:Size((jshapes[x].numel(),)) for x in jshapes}
    semiring = a.semiring
    # assert b.semiring == a.semiring
    c = MultiTensor(jshapes if transpose else ishapes, semiring)
    for x, y in a:
        axy = a[x,y].reshape(flat_ishapes[x]+flat_jshapes[y])
        if transpose:
            if x in b:
                bx = b[x].reshape(flat_ishapes[x])
                c.add_single(y, axy.T.mv(bx, semiring).reshape(jshapes[y]))
        else:
            if y in b:
                by = b[y].reshape(flat_jshapes[y])
                c.add_single(x, axy.mv(by, semiring).reshape(ishapes[x]))
    return c

def multi_solve(a: MultiTensor, b: MultiTensor, transpose: bool = False) -> MultiTensor:
    """Solve x = a @ x + b for x, where the elements of a are flattened into
    matrices and the elements of b are flattened into vectors.
    
    Arguments:
    - transpose: solve x = a.T @ x + b (or x = x @ a + b) instead.
    """

    semiring = a.semiring
    # assert b.semiring == a.semiring
    
    # Make copies of a and b flattened into matrices and vectors, respectively
    shapes = a.shapes[0]
    # assert a.shapes[0] == a.shapes[1] == b.shapes[0]
    order = list(shapes.keys())

    flat_shapes = {x:Size((shapes[x].numel(),)) for x in shapes}
    a_flat = MultiTensor((flat_shapes, flat_shapes), semiring)
    for (x, y), t in a.items():
        t = t.clone().reshape(flat_shapes[x]+flat_shapes[y])
        if transpose: 
            a_flat[y, x] = t.T
        else:
            a_flat[x, y] = t
    a = a_flat
    b_flat = MultiTensor((flat_shapes,), semiring)
    for x, t in b.items():
        b_flat[x] = t.clone().flatten()
    b = b_flat

    # LU decomposition
    for k,z in enumerate(order):
        for x in order[k+1:]:
            if (x,z) in a:
                if (z,z) in a:
                    a[x,z].copy_(a[z,z].T.solve(a[x,z].T, semiring).T)
                for y in order[k+1:]:
                    if (z,y) in a:
                        a.add_single((x,y), a[x,z].mm(a[z,y], semiring))
                if z in b:
                    b.add_single(x, a[x,z].mv(b[z], semiring))

    # Solve block-triangular systems
    for k,z in reversed(list(enumerate(order))):
        if z in b:
            if (z,z) in a:
                b[z].copy_(a[z,z].solve(b[z], semiring))
            for x in reversed(order[:k]):
                if (x,z) in a:
                    b.add_single(x, a[x,z].mv(b[z], semiring))

    # Unflatten and return solution
    b.shapes = (shapes,)
    for x in b:
        b[x] = b[x].reshape(shapes[x])
    return b
