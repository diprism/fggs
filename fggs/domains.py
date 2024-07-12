__all__ = ['Domain', 'FiniteDomain', 'RangeDomain']

from abc import ABC, abstractmethod

# this somehow needs to be able to express continuous and discrete infinite domains
# TODO: need some way to either sum over or integrate over?
class Domain(ABC):
    """Abstract base class for domains."""
    
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
        self.values = list(values)
        self._value_index = {v:i for (i,v) in enumerate(values)}

    def contains(self, value):
        return value in self.values

    def size(self):
        return len(self.values)

    def numberize(self, value):
        """Convert a value into an integer.
        Values are numbered consecutively starting from zero.
        """
        return self._value_index[value]

    def denumberize(self, num):
        """Convert a numberized value back to the original value."""
        return self.values[num]

    def __eq__(self, other):
        if self is other:
            return True
        else:
            return type(self) == type(other) and \
                   self.values == other.values

    def __ne__(self, other):
        return not self.__eq__(other)


class RangeDomain(Domain):
    """A range domain for finite sets.

    Range domain is like finite domain, but it doesn't hold
    vocabularies. Instead it only holds the size of values.

    """

    def __init__(self, size):
        super().__init__()
        self._size = size

    def size(self):
        "Return the size of the domain."
        return self._size

    def contains(self, value):
        return 0 <= value < self._size

    def numberize(self, num):
        """Convert a value into an integer.
        Since a range domain only has a size, this is actually an identify
        function.
        """
        return num

    def denumberize(self, num):
        """Convert a numberized value back to the original value."""
        return str(num)

    def __eq__(self, other):
        if self is other:
            return True
        else:
            return type(self) == type(other) and \
                   self.size() == other.size()

    def __ne__(self, other):
        return not self.__eq__(other)
