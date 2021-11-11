__all__ = ['sum_product']

from fggs.fggs import FGG, HRG, Edge, EdgeLabel
from fggs.factors import CategoricalFactor
from fggs.adjoint import adjoint_hrg
from typing import Callable, Dict, Sequence, Tuple, Iterable
from functools import reduce
import warnings, torch

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

def F(fgg: FGG, inputs: Dict[str, Tensor], nt_dict: Dict, psi_X0: Tensor) -> Tensor:
    hrg, interp = fgg.grammar, fgg.interp
    psi_X1 = psi_X0.clone()
    for nt in nt_dict:
        _, nt_shape = nt_dict[nt]
        tau_R = []
        for rule in hrg.rules(nt):
            if len(rule.rhs.nodes()) > 26:
                raise Exception('cannot assign an index to each node')
            if len(rule.rhs.edges()) == 0:
                tau_R.append(torch.ones(nt_shape))
                continue
            Xi_R = {id: chr(ord('a') + i) for i, id in enumerate(rule.rhs._node_ids)}
            indexing, tensors = [], []
            connected = set()
            for edge in rule.rhs.edges():
                indexing.append([Xi_R[node.id] for node in edge.nodes])
                connected.update(edge.nodes)
                if edge.label in inputs:
                    tensors.append(inputs[edge.label])
                elif edge.label in nt_dict:
                    (n, k), shape = nt_dict[edge.label]
                    tensors.append(psi_X0[n:k].reshape(shape))
                else:
                    raise ValueError(f'nonterminal {edge.label.name} not among inputs or outputs')
            equation = ','.join([''.join(indices) for indices in indexing]) + '->'
            # If an external node has no edges, einsum will complain, so remove it.
            external = [Xi_R[node.id] for node in rule.rhs.ext if node in connected]
            if external: equation += ''.join(external)
            tau_R_rule = torch.einsum(equation, *tensors)
            # Restore any external nodes that were removed.
            if len(external) < len(rule.rhs.ext):
                vshape = [dims if n in connected else 1 for n, dims in zip(rule.rhs.ext, nt_shape)]
                tau_R_rule = tau_R_rule.view(*vshape).expand(*nt_shape)
            tau_R.append(tau_R_rule)
        (n, k), _ = nt_dict[nt]
        psi_X1[n:k] = sum(tau_R).flatten() if len(tau_R) > 0 else torch.zeros(k - n)
    return psi_X1

def J(fgg: FGG, inputs: Dict[str, Tensor], nt_dict: Dict, psi_X0: Tensor) -> Tensor:
    hrg, interp = fgg.grammar, fgg.interp
    JF = torch.full(2 * list(psi_X0.shape), fill_value=0.)
    p, q = [0, 0], [0, 0]
    for nt_num in nt_dict:
        p[0] = p[1]
        a, b = nt_dict[nt_num][0]
        p[1] += b - a
        q_ = q[:]
        for nt_den in nt_dict:
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
                    if edge.label in inputs:
                        tensors.append(inputs[edge.label])
                    elif edge.label in nt_dict:
                        (n, k), shape = nt_dict[edge.label]
                        tensors.append(psi_X0[n:k].reshape(shape))
                    else:
                        raise ValueError(f'nonterminal {edge.label.name} not among inputs or outputs')
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

def _sum_product(fgg: FGG, opts: Dict, in_labels: Sequence[EdgeLabel], out_labels: Sequence[EdgeLabel], in_values: Sequence[Tensor]):
    """Compute the sum-product of the nonterminals in out_labels, given
    the sum-products of the terminals and/or nonterminals in in_labels
    and in_values.

    See documentation for sum_product for an explanation of the other options.

    It is an error if any rule with a LHS in out_labels has an RHS
    nonterminal not in in_labels + out_labels.
    """
    hrg, interp = fgg.grammar, fgg.interp

    inputs = dict(zip(in_labels, in_values))

    n, nt_dict = 0, {}
    for nt in out_labels:
        shape = tuple(interp.domains[node_label].size() for node_label in nt.node_labels)
        k = n + (reduce(lambda a, b: a * b, shape) if len(shape) > 0 else 1)
        nt_dict[nt] = ((n, k), shape)
        n = k
    psi_X = torch.full((n,), fill_value=0.)

    if opts['method'] == 'fixed-point':
        fixed_point(lambda psi_X: F(fgg, inputs, nt_dict, psi_X), psi_X,
                    tol=opts['tol'], kmax=opts['kmax'])
    elif opts['method'] == 'newton':
        newton(lambda psi_X: F(fgg, inputs, nt_dict, psi_X) - psi_X,
               lambda psi_X: J(fgg, inputs, nt_dict, psi_X), psi_X,
               tol=opts['tol'], kmax=opts['kmax'])
    elif opts['method'] == 'broyden':
        # Broyden's method may fail to converge without a sufficient
        # initial approximation of the Jacobian. If the method doesn't
        # converge within N iteration(s), perturb the initial approximation.
        # Source: Numerical Recipes in C: the Art of Scientific Computing
        invJ = -torch.eye(len(psi_X), len(psi_X))
        broyden(lambda psi_X: F(fgg, inputs, nt_dict, psi_X) - psi_X, invJ, psi_X,
                tol=opts['tol'], kmax=opts['kmax'])
        # k = 1
        while any(torch.isnan(psi_X)):
            # perturbation = round(random.uniform(0.51, 1.99), 1)
            psi_X = torch.full((n,), fill_value=0.)
            # invJ = -torch.eye(len(psi_X), len(psi_X)) * perturbation
            invJ = -torch.eye(len(psi_X), len(psi_X))
            broyden(lambda psi_X: F(fgg, inputs, nt_dict, psi_X) - psi_X, invJ, psi_X,
                    tol=opts['tol'], kmax=opts['kmax'])
            # broyden(lambda psi_X: F(psi_X) - psi_X, invJ, psi_X, tol=tol*(10**k), kmax=kmax)
            # k += 1
    else:
        raise ValueError('unsupported method for computing sum-product')
    out = []
    for nt in out_labels:
        (n, k), shape = nt_dict[nt]
        out.append(psi_X[n:k].reshape(shape))
    return tuple(out)

class SumProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fgg: FGG, opts: Dict, in_labels: Sequence[EdgeLabel], out_labels: Sequence[EdgeLabel], *in_values: Tensor) -> Tuple[Tensor]:
        ctx.fgg = fgg
        opts.setdefault('method', 'fixed-point')
        opts.setdefault('tol',    1e-6)
        opts.setdefault('kmax',   1000)
        ctx.opts = opts
        ctx.in_labels = in_labels
        ctx.out_labels = out_labels
        ctx.save_for_backward(*in_values)
        ctx.out_values = _sum_product(fgg, opts, in_labels, out_labels, in_values)
        return ctx.out_values

    @staticmethod
    def backward(ctx, *grad_out):
        hrg_adj, bar = adjoint_hrg(ctx.fgg.grammar) # to do: precompute or write a DF that operates directly on hrg
        fgg_adj = FGG(hrg_adj, ctx.fgg.interp)
        
        grad_in = _sum_product(fgg_adj, ctx.opts,
                               ctx.in_labels + ctx.out_labels + [bar[x] for x in ctx.out_labels],
                               [bar[x] for x in ctx.in_labels],
                               ctx.saved_tensors + ctx.out_values + grad_out)
        return (None, None, None, None) + grad_in

def sum_product(fgg: FGG, **opts) -> Tensor:
    """Compute the sum-product of an FGG.

    - fgg: The FGG to compute the sum-product of.
    - method: What method to use ('fixed-point', 'newton', 'broyden').
    - tol: Iterative algorithms terminate when the L∞ distance between consecutive iterates is below tol.
    - kmax: Number of iterations after which iterative algorithms give up.
    """
    hrg, interp = fgg.grammar, fgg.interp
    in_labels = list(hrg.terminals())
    in_values = []
    for t in in_labels:
        w = interp.factors[t]._weights
        if isinstance(w, Tensor):
            in_values.append(w)
        else:
            in_values.append(torch.tensor(w))
    out_labels = list(hrg.nonterminals())
    out = SumProduct.apply(fgg, opts, in_labels, out_labels, *in_values)
    return out[out_labels.index(fgg.grammar.start_symbol)]
