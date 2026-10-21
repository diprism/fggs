from typing import List, Tuple, Union
from itertools import chain, count
import torch
from torch import Tensor, LongTensor
from torch_semiring_einsum import Equation, compile_equation
from math import prod
from fggs.semirings import Semiring

__all__ = ['reduce_equation', 'post_einsum']


def unexpanded_shape(t : Tensor) -> Tuple[List[int], List[int], List[int]]:
    original_shape = []
    original_stride = []
    index_of_reserved_dim = []
    for idx, (strd, shap) in enumerate(zip(t.stride(), t.shape)):
        if not (strd == 0 or shap == 1):
            original_shape.append(shap)
            original_stride.append(strd)
            index_of_reserved_dim.append(idx)
    return (original_shape, original_stride, index_of_reserved_dim)


# Given
#
#   einsum(expand(E₁[x₁] → x₁y₁), ⋯, expand(Eₙ[xₙ] → xₙyₙ) → w)
#
# We have
#
#   expand(f * einsum(E₁[x₁], ⋯, Eₙ[xₙ] → u)[u] → w)
#
# where
#
# x₁ ⊥ y₁, ⋯, xₙ ⊥ yₙ
# f = ∏_{i ∈ v} |i|
# v = Y \ (X ∪ w) = Y \ X \ w = A \ w
# u = w \ (Y \ X) = w \ A
# A = Y \ X
# X = (x₁ ∪ ⋯ ∪ xₙ)
# Y = (y₁ ∪ ⋯ ∪ yₙ)
def reduce_equation(compiled_equation: Equation,
                    tensors: List[Tensor]) -> Tuple[List[Tensor],
                                                    Equation,
                                                    List[int],
                                                    List[int],
                                                    int]:
    # step 1: get the output shape, this will be used as the final expand
    output_shape = compiled_equation.get_sizes(
        tensors, compiled_equation.output_variables)

    # step 2: get the unexpanded shape of all input tensors
    shrunk_shapes, shrunk_strides, reserved_indices = zip(*[unexpanded_shape(t) for t in tensors])
    original_shapes = tuple(list(t.shape) for t in tensors)
    # return if no input tensor is an expanded tensor
    if shrunk_shapes == original_shapes:
        return (tensors, compiled_equation, [], output_shape, 1)

    # step 3: get the shrunk tensor for all input tensors
    shrunk_tensors = [torch.as_strided(t, tuple(s), tuple(d))
                      for t, s, d in zip(tensors, shrunk_shapes, shrunk_strides)]

    # step 4: reduce each input's equation
    #
    # NOTE: here we assume that no variable can occur twice in one
    # input's index, for example, the following equation works for
    # torch.einsum:
    #
    #   'ii->i'
    #
    # which takes a diagonal vector of a square matrix, but in
    # torch_semiring_einsum, it is not allowed.
    # [x₁y₁, ..., xₙyₙ]
    input_vars = compiled_equation.input_variables
    # [x₁, ..., xₙ]
    shrunk_input_vars = [[vars[idx] for idx in reserved_index]
                         for vars, reserved_index in
                         zip(input_vars, reserved_indices)]

    # step 5: collect removed input variables
    #
    # These variables will be removed from the output, so we need to
    # restore them by invoking torch.unsqueeze with the corresponding
    # dim in order
    # X = x₁ ∪ ⋯ ∪ xₙ
    shrunk_vars = set(chain(*shrunk_input_vars))
    # A = Y \ X = X ∪ Y - X
    removed_vars = set(chain(*input_vars)) - shrunk_vars
    # u = w \ A
    shrunk_out_vars = [v for v in compiled_equation.output_variables
                       if v not in removed_vars]
    # unsqueeze = w ∩ A
    unsqueeze_vars = [v for v in removed_vars
                      if v in compiled_equation.output_variables]
    unsqueeze_index = sorted([compiled_equation.output_variables.index(v)
                              for v in unsqueeze_vars])
    # v = A \ w
    factor_vars = [v for v in removed_vars
                   if v not in compiled_equation.output_variables]
    factor_shape = compiled_equation.get_sizes(tensors, factor_vars)
    factor = prod(factor_shape)

    # step 6: reduce to a new equation
    paxis_to_char = {}
    for k, c in zip(chain(shrunk_vars), map(chr, count(ord('a')))):
        paxis_to_char[k] = c

    reduced_eq_in = ','.join(''.join(paxis_to_char[dim] for dim in dims)
                             for dims in shrunk_input_vars)
    reduced_eq_out = ''.join(paxis_to_char[dim] for dim in shrunk_out_vars)
    reduced_eq = compile_equation(f'{reduced_eq_in}->{reduced_eq_out}')

    # step 7: return all information in a tuple
    return (shrunk_tensors, reduced_eq, unsqueeze_index, output_shape, factor)


def post_einsum(result: Union[Tensor, LongTensor],
                unsqueeze_index: List[int],
                output_shape: List[int],
                factor: int,
                semiring: Semiring):
    result = semiring.scale(result, factor)
    for v in unsqueeze_index:
        result = result.unsqueeze(v)
    result = result.expand(torch.Size(output_shape))
    return result


