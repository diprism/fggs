from fgg_representation import FGGRepresentation as FGG
from typing import Callable, Dict
from functools import reduce
import warnings, torch

def _formatwarning(message, category, filename=None, lineno=None, file=None, line=None):
    return '%s:%s: %s: %s' % (filename, lineno, category.__name__, message)
warnings.formatwarning = _formatwarning
Tensor = torch.Tensor; Function = Callable[[Tensor], Tensor]

def fixed_point(F: Function, psi_X0: Tensor, *, tol: float = 1e-8, maxiter: int = 1000) -> Tensor:
    psi_X1 = F(psi_X0)
    k = 0
    while any(torch.abs(psi_X1 - psi_X0) > tol) and k <= maxiter:
        psi_X0[:], psi_X1[:] = psi_X1, F(psi_X1)
        k += 1
    if k > maxiter:
        warnings.warn('maximum iteration exceeded; convergence not guaranteed')
    return psi_X1

def broyden(F: Function, J: Tensor, psi_X0: Tensor, *, tol: float = 1e-8, maxiter: int = 1000) -> Tensor:
    k = 0
    while any(torch.abs(F(psi_X0)) > tol) and k <= maxiter:
        psi_X1 = psi_X0 + torch.linalg.solve(J, -F(psi_X0))
        h = psi_X1 - psi_X0
        J += torch.einsum('i,j', (F(psi_X1) - F(psi_X0)) - torch.einsum('ij,j', J, h), h) / torch.dot(h, h)
        psi_X0[:] = psi_X1
        k += 1
    if k > maxiter:
        warnings.warn('maximum iteration exceeded; convergence not guaranteed')
    return psi_X0

def sum_product(fgg: FGG, method: str = 'fixed-point', perturbation: float = 1.0) -> Tensor:
    def get_dict(psi_X: Tensor) -> Dict[str, torch.Tensor]:
        n, nt_dict = 0, {}
        for nt_name in fgg._nonterminals:
            shape = tuple(node_label.domain.size() for node_label in fgg._nonterminals[nt_name].node_labels)
            k = n + (reduce(lambda a, b: a * b, shape) if len(shape) > 0 else 1)
            nt_dict[nt_name] = ((n, k), shape)
            n = k
        for nt_name in nt_dict:
            (n, k), shape = nt_dict[nt_name]
            nt_dict[nt_name] = psi_X[n:k].reshape(shape)
        return nt_dict
    def F(psi_X0: Tensor, soln_type: str) -> Tensor:
        psi_X1 = psi_X0.clone()
        nt_dict_1 = get_dict(psi_X1)
        nt_dict_0 = {nt_name: nt_dict_1[nt_name].clone() for nt_name in nt_dict_1}
        for nt_name in fgg._nonterminals:
            tau_R = []
            for rule in fgg.rules(nt_name):
                if len(rule.rhs().nodes()) > 52:
                    raise Exception('cannot assign an index to each node (maximum of 52 supported)')
                Xi_R = {id: chr((ord('a') if i < 26 else ord('A')) + i % 26)
                    for i, id in enumerate(rule.rhs()._node_ids)}
                indexing, tensors = [], []
                for edge in rule.rhs().edges():
                    indexing.append([Xi_R[node.id()] for node in edge.nodes()])
                    if edge.label().factor is not None:
                        weights = edge.label().factor._weights
                        tensors.append(torch.tensor(weights))
                    else:
                        tensors.append(nt_dict_0[edge.label().name])
                equation = ','.join([''.join(indices) for indices in indexing]) + '->'
                external = [Xi_R[node.id()] for node in rule.rhs().ext()]
                if external: equation += ''.join(external)
                tau_R.append(torch.einsum(equation, *tensors))
            nt_dict_1[nt_name][...] = sum(tau_R)
            if soln_type == 'zero':
                nt_dict_1[nt_name][...] -= nt_dict_0[nt_name]
        return psi_X1
    size = 0
    for nt_name in fgg._nonterminals:
        shape = tuple(node_label.domain.size() for node_label in fgg._nonterminals[nt_name].node_labels)
        size += reduce(lambda a, b: a * b, shape) if len(shape) > 0 else 1
    psi_X = torch.full((size,), fill_value=0.0)
    if method == 'fixed-point':
        psi_X = fixed_point(lambda x: F(x, soln_type='fixed-point'), psi_X)
    elif method == 'broyden':
        J = -torch.eye(len(psi_X), len(psi_X)) * perturbation
        psi_X = broyden(lambda x: F(x, soln_type='zero'), J, psi_X)
    else: raise ValueError('unsupported method for computing sum-product')
    nt_dict = get_dict(psi_X)
    return nt_dict[fgg.start_symbol().name]
