import torch
import torch_semiring_einsum # type: ignore
from abc import ABC, abstractmethod
from typing import Union

class Semiring(ABC):
    def __init__(self, dtype=torch.get_default_dtype(), device='cpu'):
        self.dtype = dtype
        self.device = device
        
    @abstractmethod
    def from_int(self, n: Union[int, torch.Tensor]):
        """Map 0 to the semiring's zero element, 1 to the semiring's one element,
        2 to 1 + 1, and so on."""
        pass
    
    @abstractmethod
    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def sub(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return any d such that x = y + d. If there is none (which isn't
        supposed to happen), just return what sub(x, x) returns."""
        pass
    
    @abstractmethod
    def mul(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def star(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ∑ xⁿ elementwise."""
        pass
    
    @abstractmethod
    def einsum(self, equation, *args: torch.Tensor, block_size: int) -> torch.Tensor:
        pass

    @abstractmethod
    def solve(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Find the least nonnegative solution of x = ax+b. Equivalently, compute ∑ aⁿb."""
        pass

class RealSemiring(Semiring):
    
    def from_int(self, n: Union[int, torch.Tensor]):
        return torch.as_tensor(n, dtype=self.dtype, device=self.device)
    
    add = staticmethod(torch.add) # type: ignore
    
    @staticmethod
    def sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.relu(x - y) # maximum(0, x-y)
    
    mul = staticmethod(torch.mul) # type: ignore
    
    @staticmethod
    def star(x: torch.Tensor) -> torch.Tensor:
        # Can't use torch.where until this is merged:
        y = 1/(1-x)
        y.masked_fill_(x >= 1, torch.inf)
        return y
        
    einsum = staticmethod(torch_semiring_einsum.einsum) # type: ignore
    
    def solve(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve(torch.eye(*a.shape, dtype=self.dtype, device=self.device)-a, b)
    
class LogSemiring(Semiring):
    
    def from_int(self, n):
        n = torch.as_tensor(n, dtype=self.dtype, device=self.device)
        return torch.log(n)
    
    add = staticmethod(torch.logaddexp) # type: ignore
    
    @staticmethod
    def sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = -torch.relu((x-y).nan_to_num()) # minimum(0, y-x)
        return x + torch.log1p(-torch.exp(z))
    
    mul = staticmethod(torch.add) # type: ignore
    
    @staticmethod
    def star(x: torch.Tensor) -> torch.Tensor:
        return -torch.log1p(-torch.torch.exp(x)).nan_to_num(nan=-torch.inf)
    
    einsum = staticmethod(torch_semiring_einsum.log_einsum) # type: ignore
    
    def solve(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Gauss-Jordan elimination / Floyd-Warshall algorithm 

        Daniel Lehmann. Algebraic structures for transitive
        closure. Theoretical Computer Science, 4(1), 1977, pages
        59-76. https://doi.org/10.1016/0304-3975(77)90056-1

        Thanks to Ryan Cotterell for pointing this out)
        """
        a = a.clone()
        x = b.clone()
        for k in range(a.shape[0]):
            a[:,k] += self.star(a[k,k])
            torch.logaddexp(a[:,k+1:], a[:,k,None] + a[k,k+1:], out=a[:,k+1:])
            torch.logaddexp(x,         a[:,k]      + x[k],      out=x)
        return x
    
class ViterbiSemiring(Semiring):
    
    def from_int(self, n):
        n = torch.as_tensor(n, device=self.device)
        return torch.where(n > 0, 0., -torch.inf).to(self.dtype)
    
    add = staticmethod(torch.maximum) # type: ignore
    
    @staticmethod
    def sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x # or torch.maximum(x, y)?
    
    mul = staticmethod(torch.add) # type: ignore
    
    @staticmethod
    def star(x: torch.Tensor) -> torch.Tensor:
        # Can't use torch.where until this is merged:
        # https://github.com/pytorch/pytorch/pull/62084
        y = x.clone()
        y.masked_fill_(x >= 0, torch.inf)
        return y
    
    @staticmethod
    def einsum(*args, **kwargs):
        val, ind = torch_semiring_einsum.log_viterbi_einsum_forward(*args, **kwargs)
        return val

    mv_equation = torch_semiring_einsum.compile_equation('ij,j->i')
    
    @staticmethod
    def solve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x = b.clone()
        for k in range(a.shape[0]):
            ax = ViterbiSemiring.einsum(ViterbiSemiring.mv_equation, a, x, block_size=10)
            torch.maximum(x, ax, out=x)
        return x
    
class BoolSemiring(Semiring):
    
    def __init__(self, device='cpu'):
        super().__init__(dtype=torch.bool, device=device)
        
    def from_int(self, n):
        n = torch.as_tensor(n, device=self.device)
        return n > 0
    
    add = staticmethod(torch.logical_or) # type: ignore
    
    @staticmethod
    def sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x & ~y
    
    mul = staticmethod(torch.logical_and) # type: ignore
    
    @staticmethod
    def star(x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def einsum(equation, *args: torch.Tensor, block_size: int) -> torch.Tensor:
        return torch_semiring_einsum.einsum(equation, *args, block_size=block_size) > 0
    
    @staticmethod
    def solve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.to(int)
        x = b.to(int)
        for k in range(a.shape[0]):
            torch.add(x, a @ x, out=x)
        return x > 0
