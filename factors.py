from abc import ABC, abstractmethod
import domains
import torch

class Factor(ABC):
    """Abstract base class for factors."""
    def __init__(self, name, doms):
        self._name = name
        self._domains = list(doms)

    def name(self):
        return self._name

    def arity(self):
        return len(self._domains)

    def domains(self):
        return self._domains
    
    @abstractmethod
    def apply(self, values):
        pass

class ConstantFactor(Factor):
    def __init__(self, name, doms, weight):
        super().__init__(name, doms)
        self._weight = weight

    def apply(self, values):
        return self._weight

class CategoricalFactor(Factor):
    def __init__(self, name, doms, weights):
        if not all(isinstance(d, domains.FiniteDomain) for d in doms):
            raise TypeError('CategoricalFactor can only be applied to FiniteDomains')
        super().__init__(name, doms)
        size = [len(d.values()) for d in doms]
        if size != list(weights.size()):
            raise ValueError(f'weight tensor has wrong size (expected {size}, actual {list(weights.size())}')
        self._weights = weights

    def weights(self):
        return self._weights

    def apply(self, values):
        """Apply factor to a sequence of values.

        values: list of strings (names)"""
        nums = tuple(d.numberize(v) for (d,v) in zip(self._domains, values))
        return self._weights[nums]
