__all__ = ['Factor', 'ConstantFactor', 'CategoricalFactor']

from abc import ABC, abstractmethod
from fggs import domains


class Factor(ABC):
    """Abstract base class for factors."""
    def __init__(self, doms):
        self._domains = tuple(doms)

    def arity(self):
        return len(self._domains)

    def domains(self):
        return self._domains
    
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
        self._weight = weight

    def apply(self, values):
        return self._weight

    def __eq__(self, other):
        if self is other:
            return True
        else:
            return type(self) == type(other) and\
                   self.domains() == other.domains() and\
                   self._weight == other._weight

        
class CategoricalFactor(Factor):
    def __init__(self, doms, weights):
        """A factor that can define an arbitrary function on finite domains.

        - doms (list of FiniteDomain): domains of arguments
        - weights (list (of lists)* of floats): weights"""
        
        if not all(isinstance(d, domains.FiniteDomain) for d in doms):
            raise TypeError('CategoricalFactor can only be applied to FiniteDomains')
        super().__init__(doms)

        def check_size(weights, size):
            if isinstance(weights, float):
                if len(size) > 0:
                    raise ValueError('weights has too few axes')
            elif isinstance(weights, list):
                if len(size) == 0:
                    raise ValueError('weights has too many axes')
                if len(weights) != size[0]:
                    raise ValueError(f'wrong number of weights (domain has {size[0]} values but weights has {len(weights)})')
                for subweights in weights:
                    check_size(subweights, size[1:])

        def to_float(weights):
            if isinstance(weights, list):
                return list(map(to_float, weights))
            elif isinstance(weights, int):
                return float(weights)
            else:
                return weights

        check_size(weights, [d.size() for d in doms])
        self._weights = to_float(weights)

    def weights(self):
        return self._weights

    def apply(self, values):
        """Apply factor to a sequence of values.

        values: list of values"""
        w = self._weights
        for d, v in zip(self._domains, values):
            w = w[d.numberize(v)]
        return w

    def __eq__(self, other):
        if self is other:
            return True
        else:
            return type(self) == type(other) and\
                   self.domains() == other.domains() and\
                   self.weights() == other.weights()
