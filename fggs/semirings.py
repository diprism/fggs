import torch
import torch_semiring_einsum
import operator

class RealSemiring:
    zero = 0.
    one = 1.
    add = operator.add
    mul = operator.mul
    einsum = torch_semiring_einsum.einsum
    from_int = lambda n: n

class LogSemiring:
    zero = -torch.inf
    one = 0.
    add = torch.logaddexp
    mul = operator.add
    einsum = torch_semiring_einsum.log_einsum
    from_int = lambda n: torch.log(n if isinstance(n, torch.Tensor) else torch.tensor(n, dtype=torch.get_default_dtype()))
    
class BoolSemiring:
    zero = False
    one = True
    add = torch.logical_or
    mul = torch.logical_and
    einsum = torch_semiring_einsum.einsum
    from_int = lambda n: (n if isinstance(n, torch.Tensor) else torch.tensor(n)) > 0
