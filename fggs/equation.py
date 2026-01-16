from typing import List, Tuple, Union
from itertools import chain, count
import torch
from torch import Tensor, LongTensor
from torch_semiring_einsum import Equation, compile_equation

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


def reduce_equation(compiled_equation: Equation,
                    tensors: List[Tensor]) -> Tuple[List[Tensor],
                                                    Equation,
                                                    List[int],
                                                    List[int]]:
    # step 1: get the output shape, this will be used as the final expand
    output_shape = compiled_equation.get_sizes(
        tensors, compiled_equation.output_variables)

    # step 2: make sure there is no sum
    # NOTE: we assume that output variable has no duplicates!
    # TODO: how to reduce equations with sum?
    if (len(compiled_equation.output_variables) != compiled_equation.num_variables):
        return (tensors, compiled_equation, [], output_shape)

    # step 3: get the unexpanded shape of all input tensors
    shrunk_shapes, shrunk_strides, reserved_indices = zip(*[unexpanded_shape(t) for t in tensors])
    original_shapes = [list(t.shape) for t in tensors]
    # return if no input tensor is an expanded tensor
    if shrunk_shapes == original_shapes:
        return (tensors, compiled_equation, [], output_shape)

    # step 4: get the shrunk tensor for all input tensors
    shrunk_tensors = [torch.as_strided(t, tuple(s), tuple(d))
                      for t, s, d in zip(tensors, shrunk_shapes, shrunk_strides)]

    # step 5: reduce each input's equation
    #
    # NOTE: here we assume that no variable can occur twice in one
    # input's index, for example, the following equation works for
    # torch.einsum:
    #
    #   'ii->i'
    #
    # which takes a diagonal vector of a square matrix, but in
    # torch_semiring_einsum, it is not allowed.
    input_vars = compiled_equation.input_variables
    shrunk_input_vars = [[vars[idx] for idx in reserved_index]
                         for vars, reserved_index in
                         zip(input_vars, reserved_indices)]

    # step 6: collect removed input variables
    #
    # These variables will be removed from the output, so we need to
    # restore them by invoking torch.unsqueeze with the corresponding
    # dim in order
    shrunk_vars = set(chain(*shrunk_input_vars))
    removed_vars = set(chain(*input_vars)) - shrunk_vars
    unsqueeze_index = sorted([compiled_equation.output_variables.index(v)
                              for v in removed_vars])
    shrunk_out_vars = [v for v in compiled_equation.output_variables
                       if v not in removed_vars]

    # step 7: reduce to a new equation
    paxis_to_char = {}
    for k, c in zip(chain(shrunk_vars), map(chr, count(ord('a')))):
        paxis_to_char[k] = c

    reduced_eq_in = ','.join(''.join(paxis_to_char[dim] for dim in dims)
                             for dims in shrunk_input_vars)
    reduced_eq_out = ''.join(paxis_to_char[dim] for dim in shrunk_out_vars)
    reduced_eq = compile_equation(f'{reduced_eq_in}->{reduced_eq_out}')

    # step 8: return all information in a tuple
    return (shrunk_tensors, reduced_eq, unsqueeze_index, output_shape)


def post_einsum(result: Union[Tensor, LongTensor],
                unsqueeze_index: List[int],
                output_shape: List[int]):
    for v in unsqueeze_index:
        result = result.unsqueeze(v)
    result = result.expand(torch.Size(output_shape))
    return result


