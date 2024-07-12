__all__ = ['Factor', 'ConstantFactor', 'FiniteFactor']

from abc import ABC, abstractmethod
from fggs import domains
from fggs.indices import PatternedTensor
import torch


class Factor(ABC):
    """Abstract base class for factors."""
    def __init__(self, doms):
        self.domains = tuple(doms)

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

    def apply(self, values):
        return self.weight

    def __eq__(self, other):
        if self is other:
            return True
        else:
            return type(self) == type(other) and \
                   self.domains == other.domains and \
                   self.weight == other.weight

        
class FiniteFactor(Factor):
    def __init__(self, doms, weights):
        """A factor that can define an arbitrary function on finite domains.

        - doms: domains of arguments
          - a list of FiniteDomain
        - weights: weights
          - a list (of lists)* of floats,
          - or a torch.Tensor,
          - or a fggs.indices.PatternedTensor"""
        
        if not all(isinstance(d, (domains.FiniteDomain, domains.RangeDomain))
                   for d in doms):
            raise TypeError('FiniteFactor can only be applied to FiniteDomains/RangeDomains')
        super().__init__(doms)

        self.weights = weights

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
