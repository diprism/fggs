from abc import ABC, abstractmethod

# this somehow needs to be able to express continuous and discrete infinite domains
# TODO: need some way to either sum over or integrate over?
class Domain(ABC):
    
    @abstractmethod
    def name(self):
        pass
    
    @abstractmethod
    def contains(self, value):
        pass


# A domain for finite sets (like vocabularies)
class FiniteDomain(Domain):
    
    def __init__(self, name, values):
        self._name = name
        self._values = values

    def name(self):
        return self._name
    
    def contains(self, value):
        return value in self._values
