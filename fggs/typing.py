from typing import TypeVar, Optional
from typing_extensions import Protocol
import torch

TensorLikeT = TypeVar('TensorLikeT', bound='TensorLike')

class TensorLike(Protocol):
    @property
    def dtype      (self: TensorLikeT) -> torch.dtype: ...
    def add        (self: TensorLikeT, other: TensorLikeT) -> TensorLikeT: ...
    def mul        (self: TensorLikeT, other: TensorLikeT) -> TensorLikeT: ...
    def logaddexp  (self: TensorLikeT, other: TensorLikeT) -> TensorLikeT: ...
    def sub        (self: TensorLikeT, other: TensorLikeT) -> TensorLikeT: ...
    def maximum    (self: TensorLikeT, other: TensorLikeT) -> TensorLikeT: ...
    def logical_and(self: TensorLikeT, other: TensorLikeT) -> TensorLikeT: ...
    def logical_or (self: TensorLikeT, other: TensorLikeT) -> TensorLikeT: ...
    def logical_not(self: TensorLikeT) -> TensorLikeT: ...
    def log_softmax(self: TensorLikeT, dim: int) -> TensorLikeT: ...
    def any        (self: TensorLikeT, dim: int, keepdim: bool = False) -> TensorLikeT: ...
    def where      (self: TensorLikeT, cond: TensorLikeT, other: TensorLikeT) -> TensorLikeT: ...
    def lt         (self: TensorLikeT, other: float) -> TensorLikeT: ...
    def le         (self: TensorLikeT, other: float) -> TensorLikeT: ...
    def gt         (self: TensorLikeT, other: float) -> TensorLikeT: ...
    def ge         (self: TensorLikeT, other: float) -> TensorLikeT: ...
    def eq         (self: TensorLikeT, other: float) -> TensorLikeT: ...
    def to         (self: TensorLikeT, dtype: torch.dtype) -> TensorLikeT: ...
    def abs        (self: TensorLikeT) -> TensorLikeT: ...
    def exp        (self: TensorLikeT) -> TensorLikeT: ...
    def expm1      (self: TensorLikeT) -> TensorLikeT: ...
    def log        (self: TensorLikeT) -> TensorLikeT: ...
    def neg_       (self: TensorLikeT) -> TensorLikeT: ...
    def log_       (self: TensorLikeT) -> TensorLikeT: ...
    def log1p_     (self: TensorLikeT) -> TensorLikeT: ...
    def relu_      (self: TensorLikeT) -> TensorLikeT: ...
    def abs_       (self: TensorLikeT) -> TensorLikeT: ...
    def nan_to_num_(self: TensorLikeT, nan: float = 0., posinf: Optional[float] = None, neginf: Optional[float] = None) -> TensorLikeT: ...
