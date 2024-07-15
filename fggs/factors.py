__all__ = ['Factor', 'ConstantFactor', 'FiniteFactor']

from abc import ABC, abstractmethod
from math import isfinite
from fggs import domains
from fggs.indices import PatternedTensor
import torch


class Factor(ABC):
    """Abstract base class for factors."""
    def __init__(self, doms):
        self.domains = tuple(doms)

    @abstractmethod
    def to_json(self):
        pass

    @property
    def arity(self):
        return len(self.domains)

    @abstractmethod
    def apply(self, values):
        pass
    
    def __eq__(self, other):
        return self is other
    
    def __ne__(self, other):
        return not self.__eq__(other)

    
class ConstantFactor(Factor):
    """A factor that always has the same weight."""
    def __init__(self, doms, weight):
        super().__init__(doms)
        self.weight = weight

    def to_json(self):
        return {'function': 'constant', 'weight': self.weight}

    def apply(self, values):
        return self.weight

    def __eq__(self, other):
        if self is other:
            return True
        else:
            return type(self) == type(other) and \
                   self.domains == other.domains and \
                   self.weight == other.weight

def weights_to_json(weights):
    if isinstance(weights, float) or hasattr(weights, 'shape') and len(weights.shape) == 0:
        return float(weights)
    else:
        return [weights_to_json(w) for w in weights]

class FiniteFactor(Factor):
    def __init__(self, doms, weights):
        """A factor that can define an arbitrary function on finite domains.

        - doms: domains of arguments
          - a list of domains whose size()s are finite
        - weights: weights
          - a list (of lists)* of floats,
          - or a torch.Tensor,
          - or a fggs.indices.PatternedTensor"""
        
        if not all(isfinite(d.size()) for d in doms):
            raise TypeError('FiniteFactor can only be applied to finite domains')
        super().__init__(doms)

        self.weights = weights

    def to_json(self):
        return {'function': 'finite', 'weights': weights_to_json(self.weights)}

    @property
    def weights(self) -> PatternedTensor:
        return self._weights

    @weights.setter
    def weights(self, weights):
        if not isinstance(weights, PatternedTensor):
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights, dtype=torch.get_default_dtype())
            weights = PatternedTensor(weights)
        size = torch.Size([d.size() for d in self.domains])
        if weights.shape != size:
            raise ValueError(f'weights has wrong shape {weights.shape} instead of expected {size}')
        self._weights = weights

    def apply(self, values):
        """Apply factor to a sequence of values.

        values: list of values"""
        return self.weights[tuple(d.numberize(v)
                                  for d, v in zip(self.domains, values))] \
                   .to_dense()

    def __eq__(self, other):
        if self is other:
            return True
        else:
            return type(self) == type(other) and \
                   self.domains == other.domains and \
                   self.weights.equal(other.weights)
