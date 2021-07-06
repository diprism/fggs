from abc import ABC, abstractmethod

# this somehow needs to be able to express continuous and discrete infinite domains
# TODO: need some way to either sum over or integrate over?
class Domain(ABC):
    
    @abstractmethod
    def contains(self, value):
        pass
    
    def __eq__(self, other):
        return self is other
    
    def __ne__(self, other):
        return not self.__eq__(other)

class FiniteDomain(Domain):
    """A domain for finite sets (like vocabularies).

    values: a collection of values, which must implement __hash__,
            __eq__, and __ne__.
    """
    
    def __init__(self, values):
        super().__init__()
        self._values = list(values)
        self._value_index = {v:i for (i,v) in enumerate(self._values)}

    def contains(self, value):
        return value in self._values

    def values(self):
        return list(self._values)

    def size(self):
        return len(self._values)

    def numberize(self, value):
        """Convert a value into an integer.
        Values are numbered consecutively starting from zero.
        """
        return self._value_index[value]

    def __eq__(self, other):
        if self is other:
            return True
        else:
            return other is not None and\
                   type(self) == type(other) and\
                   self.values() == other.values()
