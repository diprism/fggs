import torch
from math import inf
import torch_semiring_einsum, torch_semiring_einsum.utils
from abc import ABC, abstractmethod
from typing import Union

class Semiring(ABC):
    """A complete, commutative star-semiring (https://en.wikipedia.org/wiki/Semiring)."""
    def __init__(self, dtype=None, device='cpu'):
        if dtype is None: dtype = torch.get_default_dtype()
        self.dtype = dtype
        self.device = device
        
    @abstractmethod
    def from_int(self, n: Union[int, torch.Tensor]):
        """Map 0 to the semiring's zero element, 1 to the semiring's one element,
        2 to 1 + 1, and so on."""
        pass

    def eye(self, n: int) -> torch.Tensor:
        return self.from_int(torch.eye(n, dtype=self.dtype, device=self.device))
    def zeros(self, shape: torch.Size) -> torch.Tensor:
        return self.from_int(torch.zeros(shape, dtype=self.dtype, device=self.device))
    
    @abstractmethod
    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def sum(self, x: torch.Tensor, dim: int) -> torch.Tensor:
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
        """Compute x* = ∑ xⁿ = 1 + x + xx + ..., elementwise. Since 
        x* = 1 + x(x*), this lets us solve equations of the form 
        z = az+b as z = (a*)b."""
        pass
    
    @abstractmethod
    def einsum(self, equation, *args: torch.Tensor) -> torch.Tensor:
        pass
    
    def mm(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.unsqueeze(-1)
        block_size = 10
        out = self.sum(self.mul(a[:,:block_size], b[:block_size]), dim=1)
        for j in range(block_size, a.shape[1], block_size):
            out = self.add(out, self.sum(self.mul(a[:,j:j+block_size], b[j:j+block_size]), dim=1))
        return out
    
    def mv(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.sum(self.mul(a, b), dim=1)

    def solve(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Find the least nonnegative solution of x = ax+b. Equivalently, compute ∑ aⁿb.

        This is the semiring version of Gauss-Jordan elimination /
        Floyd-Warshall transitive closure.

        Daniel Lehmann. Algebraic structures for transitive
        closure. Theoretical Computer Science, 4(1), 1977, pages
        59-76. https://doi.org/10.1016/0304-3975(77)90056-1

        (Thanks to Ryan Cotterell for pointing this out.)

        """
        a = a.clone()
        x = b.clone()
        for k in range(a.shape[0]):
            a[:,k]    = self.mul(a[:,k], self.star(a[k,k]))
            a[:,k+1:] = self.add(a[:,k+1:], self.mul(a[:,k,None], a[k,k+1:]))
            if x.ndim == 1:
                x[:]  = self.add(x,         self.mul(a[:,k],      x[k]))
            elif x.ndim == 2:
                x[:]  = self.add(x,         self.mul(a[:,k,None], x[k]))
        return x

class RealSemiring(Semiring):
    
    def from_int(self, n: Union[int, torch.Tensor]):
        return torch.as_tensor(n, dtype=self.dtype, device=self.device)
    
    add = staticmethod(torch.add) # type: ignore
    sum = staticmethod(torch.sum) # type: ignore
    
    @staticmethod
    def sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # relu(x - y) = maximum(0, x - y)
        return torch.relu(x - y).nan_to_num(nan=0., posinf=inf)

    @staticmethod
    def mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mul(x, y).nan_to_num(nan=0., posinf=inf)
    
    @staticmethod
    def star(x: torch.Tensor) -> torch.Tensor:
        y = 1/(1-x)
        y.masked_fill_(x >= 1, inf)
        return y
        
    @staticmethod
    def einsum(equation, *args):
        # Make inf * 0 = 0
        def multiply_in_place(a, b):
            a.mul_(b)
            torch.nan_to_num(a, nan=0., posinf=inf, out=a)
        def callback(compute_sum):
            return compute_sum(torch_semiring_einsum.utils.add_in_place,
                               torch_semiring_einsum.utils.sum_block,
                               multiply_in_place)
        return torch_semiring_einsum.semiring_einsum_forward(equation, args, 1, callback)
    
    def solve(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # We want the least nonnegative solution of (I-a)x = b, and
        # want to use torch.linalg.solve if we can, but there are a
        # number of things that can go wrong:
        # - If a has an eigenvalue = 1, torch.linalg.solve raises RuntimeError.
        # - If a has an eigenvalue > 1, the solution will have negative components.
        # - If a has an eigenvalue = inf, the solution will have -0.0 components.
        # In these cases, we have to fall back to Semiring.solve.
        try:
            x = torch.linalg.solve(self.eye(a.shape[0])-a, b)
            if torch.all(torch.copysign(torch.tensor(1.), x) > 0.): # catches -0.0
                return x
        except RuntimeError:
            pass
        return Semiring.solve(self, a, b)


class LogSemiring(Semiring):
    
    def from_int(self, n):
        n = torch.as_tensor(n, dtype=self.dtype, device=self.device)
        return torch.log(n)
    
    add = staticmethod(torch.logaddexp) # type: ignore
    sum = staticmethod(torch.logsumexp) # type: ignore
    
    @staticmethod
    def sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # If x <= y, return -inf
        # If x ≈ y, log(exp(x) - exp(y)) = x + log(1 - exp(y-x)) = x + log(-expm1(y-x))
        # If x >> y, log(exp(x) - exp(y)) = x + log(1 - exp(y-x)) = x + log1p(-exp(y-x))
        d = y - x
        return torch.where(d < -1, # type: ignore
                           x + torch.log1p(-torch.exp(d)), # type: ignore
                           x + torch.log(-torch.expm1(d))).nan_to_num(nan=-inf, neginf=-inf, posinf=inf)
    
    @staticmethod
    def mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.add(x, y).nan_to_num(nan=-inf, neginf=-inf, posinf=inf)
    
    @staticmethod
    def star(x: torch.Tensor) -> torch.Tensor:
        # If x >= 0, return inf
        # If x ≈ 0,  log(1/(1-exp(x))) = -log(1 - exp(x)) = -log(-expm1(x))
        # If x << 0, log(1/(1-exp(x))) = -log(1 - exp(x)) = -log1p(-exp(x))
        return -torch.where(x < -1, # type: ignore
                            torch.log1p(-torch.exp(x)), # type: ignore
                            torch.log(-torch.expm1(x))).nan_to_num(nan=-inf, neginf=-inf) # type: ignore

    @staticmethod
    def einsum(equation, *args):
        
        # Slightly modified from torch_semiring_einsum/log_forward.py
        # to make log(inf) + log(0) = log(0)
        def multiply_in_place(a, b):
            a.add_(b)
            torch.nan_to_num(a, nan=-inf, posinf=inf, neginf=-inf, out=a)
        def callback(compute_sum):
            u = torch_semiring_einsum.utils
            max_values = compute_sum(u.max_in_place, u.max_block, multiply_in_place)
            u.clip_inf_in_place(max_values)
            resized_max_values = torch_semiring_einsum.log_forward.resize_max_values(
                max_values,
                equation)
            def sumexpsub_block(a, dims):
                a.sub_(resized_max_values)
                a.exp_()
                return u.sum_block(a, dims)
            result = compute_sum(u.add_in_place, sumexpsub_block, multiply_in_place)
            result.log_()
            result.add_(max_values)
            return result

        return torch_semiring_einsum.semiring_einsum_forward(equation, args, 1, callback)


class ViterbiSemiring(Semiring):
    
    def from_int(self, n):
        n = torch.as_tensor(n, device=self.device)
        return torch.where(n > 0, 0., -inf).to(self.dtype)
    
    add = staticmethod(torch.maximum) # type: ignore
    @staticmethod
    def sum(x: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.max(x, dim=dim)[0]
    
    @staticmethod
    def sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x
    
    @staticmethod
    def mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.add(x, y).nan_to_num(nan=-inf, neginf=-inf, posinf=inf)
    
    def star(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, inf, 0.).to(self.dtype)
    
    @staticmethod
    def einsum(equation, *args):
        # Make log(inf) + log(0) = log(0)
        def add_in_place(a, b):
            a.add_(b)
            torch.nan_to_num(a, nan=-inf, posinf=inf, neginf=-inf, out=a)
        def callback(compute_sum):
            return compute_sum(torch_semiring_einsum.utils.max_in_place,
                               torch_semiring_einsum.utils.max_block,
                               add_in_place,
                               include_indexes=False)
        return torch_semiring_einsum.semiring_einsum_forward(equation, args, 1, callback)

    
class BoolSemiring(Semiring):
    
    def __init__(self, device='cpu'):
        super().__init__(dtype=torch.bool, device=device)
        
    def from_int(self, n):
        n = torch.as_tensor(n, device=self.device)
        return n > 0
    
    add = staticmethod(torch.logical_or) # type: ignore
    sum = staticmethod(torch.any) # type: ignore
    
    @staticmethod
    def sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x & ~y
    
    mul = staticmethod(torch.logical_and) # type: ignore
    
    @staticmethod
    def star(x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, True)

    @staticmethod
    def einsum(equation, *args: torch.Tensor) -> torch.Tensor:
        return torch_semiring_einsum.einsum(equation, *args, block_size=1) > 0
