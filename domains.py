from abc import ABC, abstractmethod

# this somehow needs to be able to express continuous and discrete infinite domains
# TODO: need some way to either sum over or integrate over?
class Domain(ABC):
    
    def __init__(self, name):
        self._name = name
    
    def name(self):
        return self._name
    
    @abstractmethod
    def contains(self, value):
        pass

# A domain for finite sets (like vocabularies)
class FiniteDomain(Domain):
    
    def __init__(self, name, values):
        super().__init__(name)
        self._values = list(values)
        self._value_index = {v:i for (i,v) in enumerate(self._values)}

    def contains(self, value):
        return value in self._values

    def values(self):
        return self._values

    def numberize(self, value):
        return self._value_index[value]
