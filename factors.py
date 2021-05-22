from abc import ABC, abstractmethod
import domains

class Factor(ABC):
    """Abstract base class for factors."""
    def __init__(self, doms):
        self._domains = list(doms)

    def arity(self):
        return len(self._domains)

    def domains(self):
        return self._domains
    
    @abstractmethod
    def apply(self, values):
        pass

class ConstantFactor(Factor):
    def __init__(self, doms, weight):
        super().__init__(doms)
        self._weight = weight

    def apply(self, values):
        return self._weight

class CategoricalFactor(Factor):
    def __init__(self, doms, weights):
        """A factor that can define an arbitrary function on finite domains.

        doms (list of FiniteDomain): domains of arguments
        weights (list (of lists)* of floats): weights"""
        
        if not all(isinstance(d, domains.FiniteDomain) for d in doms):
            raise TypeError('CategoricalFactor can only be applied to FiniteDomains')
        super().__init__(doms)

        def check_size(weights, size):
            if not isinstance(weights, list):
                if len(size) > 0:
                    raise ValueError('weights has too few axes')
            else:
                if len(size) == 0:
                    raise ValueError('weights has too many axes')
                if len(weights) != size[0]:
                    raise ValueError(f'wrong number of weights ({size[0]} != {len(weights)})')
                for subweights in weights:
                    check_size(subweights, size[1:])

        check_size(weights, [d.size() for d in doms])
        self._weights = weights

    def weights(self):
        return self._weights

    def apply(self, values):
        """Apply factor to a sequence of values.

        values: list of values"""
        w = self._weights
        for d, v in zip(self._domains, values):
            w = w[d.numberize(v)]
        return w
