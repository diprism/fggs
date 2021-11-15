__all__ = ['sum_product']

from fggs.fggs import FGG, Interpretation, Edge, Node
from fggs.factors import CategoricalFactor
from typing import Callable, Dict, Iterable, Tuple
from functools import reduce
import warnings, torch
import collections

def _formatwarning(message, category, filename=None, lineno=None, file=None, line=None):
    return '%s:%s: %s: %s' % (filename, lineno, category.__name__, message)
warnings.formatwarning = _formatwarning
Tensor = torch.Tensor; Function = Callable[[Tensor], Tensor]

def fixed_point(F: Function, psi_X0: Tensor, *, tol: float, kmax: int) -> None:
    k, psi_X1 = 0, F(psi_X0)
    while any(torch.abs(psi_X1 - psi_X0) > tol) and k <= kmax:
        psi_X0[...], psi_X1[...] = psi_X1, F(psi_X1)
        k += 1
    if k > kmax:
        warnings.warn('maximum iteration exceeded; convergence not guaranteed')

def newton(F: Function, J: Function, psi_X0: Tensor, *, tol: float, kmax: int) -> None:
    k, psi_X1 = 0, torch.full(psi_X0.shape, fill_value=0.)
    F_X0 = F(psi_X0)
    while torch.norm(F_X0) > tol and k <= kmax:
        JF = J(psi_X0)
        dX = torch.linalg.solve(JF, -F_X0) if len(JF) > 1 else -F_X0/JF
        psi_X1[...] = psi_X0 + dX
        psi_X0[...], F_X0 = psi_X1, F(psi_X1)
        k += 1
    if k > kmax:
        warnings.warn('maximum iteration exceeded; convergence not guaranteed')

def broyden(F: Function, invJ: Tensor, psi_X0: Tensor, *, tol: float, kmax: int) -> None:
    k, psi_X1 = 0, torch.full(psi_X0.shape, fill_value=0.)
    F_X0 = F(psi_X0)
    while torch.norm(F_X0) > tol and k <= kmax:
        dX = torch.matmul(-invJ, F_X0)
        psi_X1[...] = psi_X0 + dX
        F_X1 = F(psi_X1)
        dX, dF = psi_X1 - psi_X0, F_X1 - F_X0
        u = (dX - torch.matmul(invJ, dF))/torch.dot(dF, dF)
        invJ += torch.outer(u, dF)
        psi_X0[...], F_X0 = psi_X1, F_X1
        k += 1
    if k > kmax:
        warnings.warn('maximum iteration exceeded; convergence not guaranteed')

# https://core.ac.uk/download/pdf/9821323.pdf (Computing Partition Functions of PCFGs)
# def broyden(F: Function, x: Tensor, *, tol: float = 1e-6, maxit: int = 1000, nmax: int = 20) -> None:
#     Fx = F(x); s = [Fx]
#     n, itc = -1, 0
#     while itc < maxit:
#         n, itc = n + 1, itc + 1
#         x += s[n]
#         Fx = F(x)
#         if torch.norm(Fx) <= tol:
#             break
#         if n < nmax:
#             z = Fx
#             if n > 0:
#                 for i in range(n):
#                     z += s[i + 1] * torch.dot(s[i], z)/torch.dot(s[i], s[i])
#             s.append(z/(1 - torch.dot(s[n], z)/torch.dot(s[n], s[n])))
#         elif n == nmax:
#             n, s = -1, [s[n]]
#     if itc >= maxit:
#         warnings.warn('maximum iteration exceeded; convergence not guaranteed')

def F(fgg: FGG, nt_dict: Dict[str, Tensor], psi_X0: Tensor) -> Tensor:
    hrg, interp = fgg.grammar, fgg.interp
    psi_X1 = psi_X0.clone()
    for nt in hrg.nonterminals():
        tau_R = []
        for rule in hrg.rules(nt):
            if len(rule.rhs.nodes()) > 26:
                raise Exception('cannot assign an index to each node')
            if len(rule.rhs.edges()) == 0:
                _, shape = nt_dict[nt]
                tau_R.append(torch.ones(shape))
                continue
            Xi_R = {id: chr(ord('a') + i) for i, id in enumerate(rule.rhs._node_ids)}
            indexing, tensors = [], []
            for edge in rule.rhs.edges():
                indexing.append([Xi_R[node.id] for node in edge.nodes])
                if edge.label.is_nonterminal:
                    (n, k), shape = nt_dict[edge.label]
                    tensors.append(psi_X0[n:k].reshape(shape))
                elif isinstance(interp.factors[edge.label], CategoricalFactor):
                    weights = interp.factors[edge.label]._weights
                    if not isinstance(weights, Tensor):
                        weights = torch.tensor(weights)
                    tensors.append(weights)
                else:
                    raise TypeError(f'cannot compute sum-product of FGG with factor {interp.factors[edge.label]}')
            equation = ','.join([''.join(indices) for indices in indexing]) + '->'
            external = [Xi_R[node.id] for node in rule.rhs.ext]
            if external: equation += ''.join(external)
            tau_R.append(torch.einsum(equation, *tensors))
        (n, k), _ = nt_dict[nt]
        psi_X1[n:k] = sum(tau_R).flatten() if len(tau_R) > 0 else torch.zeros(k - n)
    return psi_X1

def J(fgg: FGG, nt_dict: Dict[str, Tensor], psi_X0: Tensor) -> Tensor:
    hrg, interp = fgg.grammar, fgg.interp
    JF = torch.full(2 * list(psi_X0.shape), fill_value=0.)
    p, q = [0, 0], [0, 0]
    for nt_num in hrg.nonterminals():
        p[0] = p[1]
        a, b = nt_dict[nt_num][0]
        p[1] += b - a
        q_ = q[:]
        for nt_den in hrg.nonterminals():
            q[0] = q[1]
            a, b = nt_dict[nt_den][0]
            q[1] += b - a
            tau_R = [torch.tensor(0.)]
            for rule in hrg.rules(nt_num):
                if len(rule.rhs.nodes()) > 26:
                    raise Exception('cannot assign an index to each node')
                Xi_R = {id: chr(ord('a') + i) for i, id in enumerate(rule.rhs._node_ids)}
                nt_loc, tensors, indexing = [], [], []
                for i, edge in enumerate(rule.rhs.edges()):
                    indexing.append([Xi_R[node.id] for node in edge.nodes])
                    if edge.label.is_nonterminal:
                        (n, k), shape = nt_dict[edge.label]
                        tensors.append(psi_X0[n:k].reshape(shape))
                        nt_loc.append((i, edge.label.name))
                    elif isinstance(interp.factors[edge.label], CategoricalFactor):
                        weights = interp.factors[edge.label]._weights
                        if not isinstance(weights, Tensor):
                            weights = torch.tensor(weights)
                        tensors.append(weights)
                    else:
                        raise TypeError(f'cannot compute sum-product of FGG with factor {interp.factors[edge.label]}')
                external = [Xi_R[node.id] for node in rule.rhs.ext]
                alphabet = (chr(ord('a') + i) for i in range(26))
                indices = set(x for sublist in indexing for x in sublist)
                diff_index = next(x for x in alphabet if x not in indices) if indexing[i] else ''
                x = [torch.tensor(0.)]
                for i, nt_name in nt_loc:
                    if nt_den.name != nt_name:
                        continue
                    if len(tensors) > 1:
                        equation = ','.join(''.join(indices) for j, indices in enumerate(indexing) if j != i) + '->'
                        if external: equation += ''.join(external)
                        if diff_index:
                            equation = equation.replace(''.join(indexing[i]), diff_index) + diff_index
                        x.append(torch.einsum(equation, *(tensor for j, tensor in enumerate(tensors) if j != i)))
                    else:
                        x.append(torch.ones(tensors[i].size()))
                tau_R.append(sum(t.sum() for t in x))
            x = sum(t.sum() for t in tau_R)
            if nt_num.name == nt_den.name:
                x -= 1 if x.size() == torch.Size([]) else torch.eye(x.size())
            JF[p[0]:p[1], q[0]:q[1]] = x
        q = q_[:]
    return JF

def sum_product_edges(interp: Interpretation, ext: Tuple[Node], edges: Iterable[Edge]) -> Tensor:
    # Each node corresponds to an index, so choose a letter for each
    nodes = set()
    for edge in edges:
        nodes.update(edge.nodes)
    if len(nodes) > 26:
        raise Exception('cannot assign an index to each node')
    node_to_letter = {node: chr(ord('a') + i) for i, node in enumerate(nodes)}
    
    indexing, tensors = [], []
    for edge in edges:
        indexing.append([node_to_letter[node] for node in edge.nodes])
        if edge.label.is_nonterminal:
            raise NotImplementedError()
        elif isinstance(interp.factors[edge.label], CategoricalFactor):
            weights = interp.factors[edge.label]._weights
            if not isinstance(weights, Tensor):
                weights = torch.tensor(weights)
            tensors.append(weights)
        else:
            raise TypeError(f'cannot compute sum-product of FGG with factor {interp.factors[edge.label]}')
    equation = ','.join([''.join(indices) for indices in indexing]) + '->'
    
    # If an external node has no edges, einsum will complain, so remove it.
    external = [node_to_letter[node] for node in ext if node in nodes]
    equation += ''.join(external)
    
    out = torch.einsum(equation, *tensors)
    
    # Restore any external nodes that were removed.
    if len(external) < len(nodes):
        vshape = [interp.domains[n.label].size() if n in nodes else 1 for n in ext]
        eshape = [interp.domains[n.label].size() for n in ext]
        out = out.view(*vshape).expand(*eshape)
        
    return out

def linear(fgg: FGG, nt_dict: Dict, n: int) -> Tensor:
    hrg, interp = fgg.grammar, fgg.interp

    nullary_index = {x:[] for x in hrg.nonterminals()}
    unary_index = {(x,y):[] for x in hrg.nonterminals() for y in hrg.nonterminals()}
    for x in hrg.nonterminals():
        for rule in hrg.rules(x):
            edges = [e for e in rule.rhs.edges() if e.label.is_nonterminal]
            if len(edges) == 0:
                nullary_index[x].append(rule)
            elif len(edges) == 1:
                unary_index[x, edges[0].label].append((rule, edges[0]))
            else:
                raise ValueError('FGG is not linearly recursive')

    f = torch.zeros(n)
    jf = torch.zeros(n, n)
    for x in hrg.nonterminals():
        (xi, xj), _ = nt_dict[x]
        # Compute JF(0)
        for y in hrg.nonterminals():
            (yi, yj), _ = nt_dict[y]
            z_rules = []
            for rule, edge in unary_index[x, y]:
                ext = rule.rhs.ext + edge.nodes
                edges = set(rule.rhs.edges()) - {edge}
                z_rules.append(sum_product_edges(interp, ext, edges))
            if len(z_rules) > 0:
                jf[xi:xj,yi:yj] = sum(z_rules).view(xj-xi, yj-yi)

        # Compute F(0)
        z_rules = []
        for rule in nullary_index[x]:
            z_rules.append(sum_product_edges(interp, rule.rhs.ext, rule.rhs.edges()))
        if len(z_rules) > 0:
            f[xi:xj] = sum(z_rules).view(xj-xi)

    return torch.linalg.solve(torch.eye(n)-jf, f)

def sum_product(fgg: FGG, *, method: str = 'fixed-point', tol: float = 1e-6, kmax: int = 1000) -> Tensor:
    """Compute the sum-product of an FGG.

    - fgg: The FGG to compute the sum-product of.
    - method: What method to use ('linear', 'fixed-point', 'newton', 'broyden').
    - tol: Iterative algorithms terminate when the L∞ distance between consecutive iterates is below tol.
    - kmax: Number of iterations after which iterative algorithms give up.
    """
    hrg, interp = fgg.grammar, fgg.interp
    n, nt_dict = 0, {}
    for nt in hrg.nonterminals():
        shape = tuple(interp.domains[node_label].size() for node_label in nt.node_labels)
        k = n + (reduce(lambda a, b: a * b, shape) if len(shape) > 0 else 1)
        nt_dict[nt] = ((n, k), shape)
        n = k
    psi_X = torch.full((n,), fill_value=0.)
    if method == 'linear':
        psi_X = linear(fgg, nt_dict, n)
    elif method == 'fixed-point':
        fixed_point(lambda psi_X: F(fgg, nt_dict, psi_X), psi_X, tol=tol, kmax=kmax)
    elif method == 'newton':
        newton(lambda psi_X: F(fgg, nt_dict, psi_X) - psi_X, lambda psi_X: J(fgg, nt_dict, psi_X), psi_X, tol=tol, kmax=kmax)
    elif method == 'broyden':
        # Broyden's method may fail to converge without a sufficient
        # initial approximation of the Jacobian. If the method doesn't
        # converge within N iteration(s), perturb the initial approximation.
        # Source: Numerical Recipes in C: the Art of Scientific Computing
        invJ = -torch.eye(len(psi_X), len(psi_X))
        broyden(lambda psi_X: F(fgg, nt_dict, psi_X) - psi_X, invJ, psi_X, tol=tol, kmax=kmax)
        # k = 1
        while any(torch.isnan(psi_X)):
            # perturbation = round(random.uniform(0.51, 1.99), 1)
            psi_X = torch.full((n,), fill_value=0.)
            # invJ = -torch.eye(len(psi_X), len(psi_X)) * perturbation
            invJ = -torch.eye(len(psi_X), len(psi_X))
            broyden(lambda psi_X: F(fgg, nt_dict, psi_X) - psi_X, invJ, psi_X, tol=tol, kmax=kmax)
            # broyden(lambda psi_X: F(psi_X) - psi_X, invJ, psi_X, tol=tol*(10**k), kmax=kmax)
            # k += 1
    else: raise ValueError('unsupported method for computing sum-product')
    (n, k), shape = nt_dict[hrg.start_symbol]
    return psi_X[n:k].reshape(shape)
