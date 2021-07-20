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

def broyden(F: Function, invJ: Tensor, psi_X0: Tensor, *, tol: float = 1e-8, maxiter: int = 1000) -> None:
    psi_X1 = torch.full(psi_X0.shape, fill_value=0.0)
    F_X0 = F(psi_X0)
    k = 0
    while any(torch.abs(F_X0) > tol) and k <= maxiter:
        psi_X1[...] = psi_X0 + torch.matmul(-invJ, F_X0)
        F_X1 = F(psi_X1)
        dx, df = psi_X1 - psi_X0, F_X1 - F_X0
        v = torch.matmul(torch.transpose(-invJ, 0, 1), dx)
        u = (dx - torch.matmul(invJ, df))/torch.dot(v, df)
        invJ += torch.outer(u, v)
        psi_X0[...], F_X0 = psi_X1, F_X1
        k += 1
    if k > maxiter:
        warnings.warn('maximum iteration exceeded; convergence not guaranteed')

def sum_product(fgg: FGG, method: str = 'fixed-point', perturbation: float = 1.0) -> Tensor:
    n, nt_dict = 0, {}
    for nt in fgg.nonterminals():
        shape = tuple(node_label.domain.size() for node_label in nt.node_labels)
        k = n + (reduce(lambda a, b: a * b, shape) if len(shape) > 0 else 1)
        nt_dict[nt] = ((n, k), shape)
        n = k
    psi_X = torch.full((n,), fill_value=0.0)
    def F(psi_X0: Tensor) -> Tensor:
        psi_X1 = psi_X0.clone()
        for nt in fgg.nonterminals():
            tau_R = []
            for rule in fgg.rules(nt):
                if len(rule.rhs().nodes()) > 52:
                    raise Exception('cannot assign an index to each node (maximum of 52 supported)')
                Xi_R = {id: chr((ord('a') if i < 26 else ord('A')) + i % 26)
                    for i, id in enumerate(rule.rhs()._node_ids)}
                indexing, tensors = [], []
                for edge in rule.rhs().edges():
                    indexing.append([Xi_R[node.id] for node in edge.nodes])
                    if edge.label.is_nonterminal():
                        (n, k), shape = nt_dict[edge.label]
                        tensors.append(psi_X0[n:k].reshape(shape))
                    elif isinstance(edge.label.factor, CategoricalFactor):
                        weights = edge.label.factor._weights
                        tensors.append(torch.tensor(weights))
                    else:
                        raise TypeError(f"Can't compute sum-product of FGG with factor {edge.label.factor}")
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
        invJ = -torch.eye(len(psi_X), len(psi_X)) * perturbation
        broyden(lambda psi_X: F(psi_X) - psi_X, invJ, psi_X)
    else: raise ValueError('unsupported method for computing sum-product')
    (n, k), shape = nt_dict[fgg.start_symbol()]
    return psi_X[n:k].reshape(shape)
