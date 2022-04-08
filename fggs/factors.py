__all__ = ['Factor', 'ConstantFactor', 'FiniteFactor']

from abc import ABC, abstractmethod
from fggs import domains


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

        - doms (list of FiniteDomain): domains of arguments
        - weights (list (of lists)* of floats): weights"""
        
        if not all(isinstance(d, domains.FiniteDomain) for d in doms):
            raise TypeError('FiniteFactor can only be applied to FiniteDomains')
        super().__init__(doms)

        def check_size(weights, size):
            if hasattr(weights, 'shape'):
                return weights.shape == size
            elif isinstance(weights, (int, float)):
                if len(size) > 0:
                    raise ValueError('weights has too few axes')
            elif isinstance(weights, (list, tuple)):
                if len(size) == 0:
                    raise ValueError('weights has too many axes')
                if len(weights) != size[0]:
                    raise ValueError(f'wrong number of weights (domain has {size[0]} values but weights has {len(weights)})')
                for subweights in weights:
                    check_size(subweights, size[1:])
            else:
                raise TypeError(f"weights are wrong type ({type(weights)})")

        check_size(weights, [d.size() for d in doms])
        self.weights = weights

    def apply(self, values):
        """Apply factor to a sequence of values.

        values: list of values"""
        w = self.weights
        for d, v in zip(self.domains, values):
            w = w[d.numberize(v)]
        return w

    def __eq__(self, other):
        if self is other:
            return True
        else:
            return type(self) == type(other) and \
                   self.domains == other.domains and \
                   self.weights == other.weights
