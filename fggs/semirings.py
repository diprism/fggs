import torch
import torch_semiring_einsum # type: ignore
import operator

class RealSemiring:
    zero = 0.
    one = 1.
    add = staticmethod(operator.add)
    mul = staticmethod(operator.mul)
    einsum = staticmethod(torch_semiring_einsum.einsum)
    from_int = staticmethod(lambda n: n)

class LogSemiring:
    zero = -torch.inf
    one = 0.
    add = staticmethod(torch.logaddexp)
    mul = staticmethod(operator.add)
    einsum = staticmethod(torch_semiring_einsum.log_einsum)
    @staticmethod
    def from_int(n):
        if not isinstance(n, torch.Tensor):
            n = torch.tensor(n, dtype=torch.get_default_dtype())
        return torch.log(n)
    
class BoolSemiring:
    zero = False
    one = True
    add = staticmethod(torch.logical_or)
    mul = staticmethod(torch.logical_and)
    einsum = staticmethod(torch_semiring_einsum.einsum)
    @staticmethod
    def from_int(n):
        if not isinstance(n, torch.Tensor):
            n = torch.tensor(n)
        return n > 0
