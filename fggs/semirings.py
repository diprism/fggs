import torch
import torch_semiring_einsum # type: ignore
from abc import ABC, abstractmethod
from typing import Union

class Semiring(ABC):
    def __init__(self, dtype=torch.get_default_dtype(), device='cpu'):
        self.dtype = dtype
        self.device = device
        
    def from_int(self, n: Union[int, torch.Tensor]):
        return torch.as_tensor(n, dtype=self.dtype, device=self.device)
    
    @staticmethod
    @abstractmethod
    def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass
    
    @staticmethod
    @abstractmethod
    def mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass
    
    @staticmethod
    @abstractmethod
    def einsum(equation, *args: torch.Tensor, block_size: int) -> torch.Tensor:
        pass


class RealSemiring(Semiring):
    add = staticmethod(torch.add) # type: ignore
    mul = staticmethod(torch.mul) # type: ignore
    einsum = staticmethod(torch_semiring_einsum.einsum) # type: ignore
    

class LogSemiring(Semiring):
    def from_int(self, n):
        return torch.log(super().from_int(n))
    add = staticmethod(torch.logaddexp) # type: ignore
    mul = staticmethod(torch.add) # type: ignore
    einsum = staticmethod(torch_semiring_einsum.log_einsum) # type: ignore


class ViterbiSemiring(Semiring):
    def from_int(self, n):
        return torch.where(torch.as_tensor(n, device=self.device) > 0, 0., -torch.inf)
    add = staticmethod(torch.maximum) # type: ignore
    mul = staticmethod(torch.add) # type: ignore
    @staticmethod
    def einsum(*args, **kwargs):
        try:
            val, ind = torch_semiring_einsum.log_viterbi_einsum_forward(*args, **kwargs)
        except:
            import pdb
            pdb.set_trace()
        return val

    
class BoolSemiring(Semiring):
    def __init__(self, device='cpu'):
        super().__init__(dtype=torch.bool, device=device)
    add = staticmethod(torch.logical_or) # type: ignore
    mul = staticmethod(torch.logical_and) # type: ignore
    einsum = staticmethod(torch_semiring_einsum.einsum) # type: ignore
