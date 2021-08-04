__all__ = ['sum_product']

from fggs.fggs import FGG
from fggs.factors import CategoricalFactor
from typing import Callable
from functools import reduce
import warnings, torch

def _formatwarning(message, category, filename=None, lineno=None, file=None, line=None):
    return '%s:%s: %s: %s' % (filename, lineno, category.__name__, message)
warnings.formatwarning = _formatwarning
Tensor = torch.Tensor; Function = Callable[[Tensor], Tensor]

def fixed_point(F: Function, psi_X0: Tensor, *, tol: float = 1e-8, maxiter: int = 1000) -> None:
    psi_X1 = F(psi_X0)
    k = 0
    while any(torch.abs(psi_X1 - psi_X0) > tol) and k <= maxiter:
        psi_X0[...], psi_X1[...] = psi_X1, F(psi_X1)
        k += 1
    if k > maxiter:
        warnings.warn('maximum iteration exceeded; convergence not guaranteed')

def broyden(F: Function, J: Tensor, psi_X0: Tensor, *, tol: float = 1e-8, maxiter: int = 1000) -> None:
    psi_X1 = torch.full(psi_X0.shape, fill_value=0.0)
    F_X0 = F(psi_X0)
    k = 0
    while any(torch.abs(F_X0) > tol) and k <= maxiter:
        psi_X1[...] = psi_X0 + torch.linalg.solve(J, -F_X0)
        h = psi_X1 - psi_X0
        J += torch.outer((F(psi_X1) - F(psi_X0)) - torch.matmul(J, h), h) / torch.dot(h, h)
        psi_X0[...], F_X0 = psi_X1, F(psi_X1)
        k += 1
    if k > maxiter:
        warnings.warn('maximum iteration exceeded; convergence not guaranteed')

def sum_product(fgg: FGG, method: str = 'fixed-point', perturbation: float = 1.0) -> Tensor:
    hrg, interp = fgg.grammar, fgg.interp
    n, nt_dict = 0, {}
    for nt in hrg.nonterminals():
        shape = tuple(interp.domains[node_label].size() for node_label in nt.node_labels)
        k = n + (reduce(lambda a, b: a * b, shape) if len(shape) > 0 else 1)
        nt_dict[nt] = ((n, k), shape)
        n = k
    psi_X = torch.full((n,), fill_value=0.0)
    def F(psi_X0: Tensor) -> Tensor:
        psi_X1 = psi_X0.clone()
        for nt in hrg.nonterminals():
            tau_R = []
            for rule in hrg.rules(nt):
                if len(rule.rhs().nodes()) > 52:
                    raise Exception('cannot assign an index to each node (maximum of 52 supported)')
                Xi_R = {id: chr((ord('a') if i < 26 else ord('A')) + i % 26)
                    for i, id in enumerate(rule.rhs()._node_ids)}
                indexing, tensors = [], []
                for edge in rule.rhs().edges():
                    indexing.append([Xi_R[node.id] for node in edge.nodes])
                    if edge.label.is_nonterminal:
                        (n, k), shape = nt_dict[edge.label]
                        tensors.append(psi_X0[n:k].reshape(shape))
                    elif isinstance(interp.factors[edge.label], CategoricalFactor):
                        weights = interp.factors[edge.label]._weights
                        tensors.append(torch.tensor(weights))
                    else:
                        raise TypeError(f"Can't compute sum-product of HRG with factor {interp.factors[edge.label]}")
                equation = ','.join([''.join(indices) for indices in indexing]) + '->'
                external = [Xi_R[node.id] for node in rule.rhs().ext()]
                if external: equation += ''.join(external)
                tau_R.append(torch.einsum(equation, *tensors))
            (n, k), _ = nt_dict[nt]
            psi_X1[n:k] = sum(tau_R).flatten()
        return psi_X1
    if method == 'fixed-point':
        fixed_point(F, psi_X)
    elif method == 'broyden':
        # Broyden's method may fail to converge without a sufficient
        # initial approximation of the Jacobian. If the method doesn't
        # converge within N iteration(s), perturb the initial approximation.
        # Source: Numerical Recipes in C: the Art of Scientific Computing
        J = -torch.eye(len(psi_X), len(psi_X)) * perturbation
        broyden(lambda psi_X: F(psi_X) - psi_X, J, psi_X)
    else: raise ValueError('unsupported method for computing sum-product')
    (n, k), shape = nt_dict[hrg.start_symbol()]
    return psi_X[n:k].reshape(shape)
