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

    def solve(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Find the least nonnegative solution of x = ax+b. Equivalently, compute ∑ aⁿb.

        This is the semiring version of Gauss-Jordan elimination /
        Floyd-Warshall transitive closure..

        Daniel Lehmann. Algebraic structures for transitive
        closure. Theoretical Computer Science, 4(1), 1977, pages
        59-76. https://doi.org/10.1016/0304-3975(77)90056-1

        Thanks to Ryan Cotterell for pointing this out)

        """
        a = a.clone()
        x = b.clone()
        for k in range(a.shape[0]):
            a[:,k]    = self.mul(a[:,k], self.star(a[k,k]))
            a[:,k+1:] = self.add(a[:,k+1:], self.mul(a[:,k,None], a[k,k+1:]))
            x[:]      = self.add(x,         self.mul(a[:,k],      x[k]))
        return x

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
        y = 1/(1-x)
        y.masked_fill_(x >= 1, torch.inf)
        return y
        
    einsum = staticmethod(torch_semiring_einsum.einsum) # type: ignore
    
    def solve(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x = torch.linalg.solve(torch.eye(*a.shape, dtype=self.dtype, device=self.device)-a, b)
        # We want to find the least nonnegative solution of (I-a)x = b, so check
        # that all components are nonnegative.
        if torch.any(x < 0) or torch.any(x.isnan()):
            # There is no (finite) solution. Fall back to Semiring.solve, which can return inf.
            x = Semiring.solve(self, a, b)
        return x
    
class LogSemiring(Semiring):
    
    def from_int(self, n):
        n = torch.as_tensor(n, dtype=self.dtype, device=self.device)
        return torch.log(n)
    
    add = staticmethod(torch.logaddexp) # type: ignore
    
    @staticmethod
    def sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = -torch.relu((x-y).nan_to_num()) # type: ignore # minimum(0, y-x)
        return x - LogSemiring.star(z)
    
    mul = staticmethod(torch.add) # type: ignore
    
    @staticmethod
    def star(x: torch.Tensor) -> torch.Tensor:
        return -torch.where(x < -1,
                            torch.log1p(-torch.exp(x)),
                            torch.log(-torch.expm1(x))).nan_to_num(nan=-torch.inf) # type: ignore
    
    einsum = staticmethod(torch_semiring_einsum.log_einsum) # type: ignore
    
    
class ViterbiSemiring(Semiring):
    
    def from_int(self, n):
        n = torch.as_tensor(n, device=self.device)
        return torch.where(n > 0, 0., -torch.inf).to(self.dtype)
    
    add = staticmethod(torch.maximum) # type: ignore
    
    @staticmethod
    def sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x # or torch.maximum(x, y)?
    
    mul = staticmethod(torch.add) # type: ignore
    
    def star(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, torch.inf, 0.).to(self.dtype)
    
    @staticmethod
    def einsum(*args, **kwargs):
        val, ind = torch_semiring_einsum.log_viterbi_einsum_forward(*args, **kwargs)
        return val

    mv_equation = torch_semiring_einsum.compile_equation('ij,j->i')
    
    
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
        return torch.full_like(x, True)

    @staticmethod
    def einsum(equation, *args: torch.Tensor, block_size: int) -> torch.Tensor:
        return torch_semiring_einsum.einsum(equation, *args, block_size=block_size) > 0
