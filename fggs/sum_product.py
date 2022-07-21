__all__ = ['sum_product', 'sum_products']

from fggs.fggs import FGG, HRG, HRGRule, EdgeLabel, Edge, Node
from fggs.domains import FiniteDomain
from fggs.factors import FiniteFactor
from fggs.semirings import *
from fggs.multi import *
from fggs.utils import scc, nonterminal_graph
from math import inf

from typing import Callable, Dict, Mapping, Sequence, Iterable, Tuple, List, Set, Union, Optional, cast
import warnings

import torch
from torch import Tensor
import torch_semiring_einsum

Function = Callable[[MultiTensor], MultiTensor]

def _formatwarning(message, category, filename=None, lineno=None, file=None, line=None):
    return '%s:%s: %s: %s' % (filename, lineno, category.__name__, message)
warnings.formatwarning = _formatwarning # type: ignore


class FGGMultiShape(MultiShape):
    """A virtual MultiShape for the nonterminals or terminals of an FGG."""
    def __init__(self, fgg, els):
        self.fgg = fgg
        self.els = list(els)
        self.elset = set(els)
    def __getitem__(self, x):
        if x in self.elset:
            return torch.Size(self.fgg.shape(x))
        else:
            raise KeyError()
    def __iter__(self):
        return iter(self.els)
    def __len__(self):
        return len(self.els)
    def __str__(self):
        return str(dict(self))
    def __repr__(self):
        return repr(dict(self))

    
def fixed_point(F: Function, x0: MultiTensor, *, tol: float, kmax: int) -> None:
    """Fixed-point iteration method for solving x = F(x)."""
    k, x1 = 0, F(x0)
    while not x0.allclose(x1, tol) and k <= kmax:
        x0.copy_(x1)
        x1.copy_(F(x1))
        k += 1
    if k > kmax:
        warnings.warn('maximum iteration exceeded; convergence not guaranteed')


def newton(F: Function, J: Function, x0: MultiTensor, *, tol: float, kmax: int) -> None:
    """Newton's method for solving x = F(x) in a commutative semiring.

    Javier Esparza, Stefan Kiefer, and Michael Luttenberger. On fixed
    point equations over commutative semirings. In Proc. STACS, 2007."""
    semiring = x0.semiring
    x1 = MultiTensor(x0.shapes, x0.semiring)
    for k in range(kmax):
        F0 = F(x0)
        if F0.allclose(x0, tol): break
        JF = J(x0)
        dX = multi_solve(JF, F0 - x0)
        x0.copy_(x0 + dX)

    if k > kmax:
        warnings.warn('maximum iteration exceeded; convergence not guaranteed')


def F(fgg: FGG, x: MultiTensor, inputs: MultiTensor, semiring: Semiring) -> MultiTensor:
    Fx = MultiTensor(x.shapes, x.semiring)
    for n in x.shapes[0]:
        for rule in fgg.rules(n):
            tau_rule = sum_product_edges(fgg, rule.rhs.nodes(), rule.rhs.edges(), rule.rhs.ext, x, inputs, semiring=semiring)
            Fx.add_single(n, tau_rule)
    return Fx


def J(fgg: FGG, x: MultiTensor, inputs: MultiTensor, semiring: Semiring,
      J_inputs: Optional[MultiTensor] = None) -> MultiTensor:
    """The Jacobian of F."""
    Jx = MultiTensor(x.shapes+x.shapes, semiring)
    for n in x.shapes[0]:
        for rule in fgg.rules(n):
            for edge in rule.rhs.edges():
                if edge.label not in Jx.shapes[1] and J_inputs is None: continue
                ext = rule.rhs.ext + edge.nodes
                edges = set(rule.rhs.edges()) - {edge}
                tau_edge = sum_product_edges(fgg, rule.rhs.nodes(), edges, ext, x, inputs, semiring=semiring)
                if edge.label in Jx.shapes[1]:
                    Jx.add_single((n, edge.label), tau_edge)
                elif J_inputs is not None and edge.label in J_inputs.shapes[1]:
                    J_inputs.add_single((n, edge.label), tau_edge)
                else:
                    assert False
    return Jx


def log_softmax(a: Tensor, dim: int) -> Tensor:
    # If a has infinite elements, log_softmax would return all nans.
    # In this case, make all the nonzero elements 1 and the zero elements 0.
    return torch.where(torch.any(a == inf, dim, keepdim=True),
                       torch.log(a > -inf).to(dtype=a.dtype),
                       torch.log_softmax(a, dim))
    
def J_log(fgg: FGG, x: MultiTensor, inputs: MultiTensor, semiring: Semiring,
          J_inputs: Optional[MultiTensor] = None) -> MultiTensor:
    """The Jacobian of F(semiring=LogSemiring), computed in the real semiring."""
    Jx = MultiTensor(x.shapes+x.shapes, semiring=RealSemiring(dtype=semiring.dtype, device=semiring.device))
    for n in x.shapes[0]:
        rules = list(fgg.rules(n))
        tau_rules: Union[List[Tensor], Tensor]
        tau_rules = []
        for rule in rules:
            tau_rule = sum_product_edges(fgg, rule.rhs.nodes(), rule.rhs.edges(), rule.rhs.ext, x, inputs, semiring=semiring)
            tau_rules.append(tau_rule)
        if len(tau_rules) == 0: continue
        tau_rules = torch.stack(tau_rules, dim=0)
        tau_rules = log_softmax(tau_rules, dim=0)
        for rule, tau_rule in zip(rules, tau_rules):
            for edge in rule.rhs.edges():
                if edge.label not in Jx.shapes[1] and J_inputs is None: continue
                ext = rule.rhs.ext + edge.nodes
                tau_edge = sum_product_edges(fgg, rule.rhs.nodes(), rule.rhs.edges(), ext, x, inputs, semiring=semiring)
                tau_edge_size = tau_edge.size()
                tau_edge = tau_edge.reshape(tau_rule.size() + (-1,))
                tau_edge = log_softmax(tau_edge, dim=-1)
                tau_edge += tau_rule.unsqueeze(-1)
                tau_edge = tau_edge.reshape(tau_edge_size)
                if edge.label in Jx.shapes[1]:
                    Jx.add_single((n, edge.label), torch.exp(tau_edge))
                elif J_inputs is not None and edge.label in J_inputs.shapes[1]:
                    J_inputs.add_single((n, edge.label), torch.exp(tau_edge))
                else:
                    assert False
    return Jx


def sum_product_edges(fgg: FGG, nodes: Iterable[Node], edges: Iterable[Edge], ext: Sequence[Node], *inputses: MultiTensor, semiring: Semiring) -> Tensor:
    """Compute the sum-product of a set of edges.

    Parameters:
    - fgg
    - ext: the nodes whose values are not summed over
    - edges: the edges whose factors are multiplied together
    - inputses: dicts of sum-products of nonterminals that have
      already been computed. Later elements of inputses override
      earlier elements.

    Return: the tensor of sum-products
    """

    connected: Set[Node] = set()
    indexing: List[Iterable[Node]] = []
    tensors: List[Tensor] = []

    # Derivatives can sometimes produce duplicate external nodes.
    # Rename them apart and add identity factors between them.
    ext_orig = ext
    ext = []
    for n in ext_orig:
        if n in ext:
            ncopy = Node(n.label)
            ext.append(ncopy)
            connected.update([n, ncopy])
            indexing.append([n, ncopy])
            nsize = cast(FiniteDomain, fgg.domains[n.label.name]).size()
            tensors.append(semiring.eye(nsize))
        else:
            ext.append(n)

    for edge in edges:
        connected.update(edge.nodes)
        indexing.append(edge.nodes)
        for inputs in reversed(inputses):
            if edge.label in inputs:
                tensors.append(inputs[edge.label])
                break
        else:
            # One argument to einsum will be the zero tensor, so just return zero
            return semiring.zeros(fgg.shape(ext))

    if len(indexing) > 0:
        # Each node corresponds to an index, so choose a letter for each
        if len(connected) > 26:
            raise Exception('cannot assign an index to each node')
        node_to_index = {node: chr(ord('a') + i) for i, node in enumerate(connected)}

        equation = ','.join([''.join(node_to_index[n] for n in indices) for indices in indexing]) + '->'

        # If an external node has no edges, einsum will complain, so remove it.
        equation += ''.join(node_to_index[node] for node in ext if node in connected)

        compiled = torch_semiring_einsum.compile_equation(equation)
        out = semiring.einsum(compiled, *tensors)
    else:
        out = semiring.from_int(1)

    # Restore any external nodes that were removed.
    if out.ndim < len(ext):
        eshape = fgg.shape(ext)
        vshape = [s if n in connected else 1 for n, s in zip(ext, eshape)]
        rshape = [1 if n in connected else s for n, s in zip(ext, eshape)]
        out = out.view(*vshape).repeat(*rshape)

    # Multiply in any disconnected internal nodes.
    mul = 1
    for n in nodes:
        if n not in connected and n not in ext:
            mul *= cast(FiniteDomain, fgg.domains[n.label.name]).size()
    if mul > 1:
        out = semiring.mul(out, semiring.from_int(mul))
    return out


def linear(fgg: FGG, inputs: MultiTensor, out_labels: Sequence[EdgeLabel], semiring: Semiring) -> MultiTensor:
    """Compute the sum-product of the nonterminals of `fgg`, which is
    linearly recursive given that each nonterminal `x` in `inputs` is
    treated as a terminal with weight `inputs[x]`.
    """
    shapes = FGGMultiShape(fgg, out_labels)

    # Check linearity and compute F(0) and J(0)
    F0 = MultiTensor(shapes, semiring)
    J0 = MultiTensor((shapes, shapes), semiring)
    for n in out_labels:
        if n in inputs:
            continue
        for rule in fgg.rules(n):
            edges = [e for e in rule.rhs.edges() if e.label not in inputs]
            if len(edges) == 0:
                F0.add_single(n, sum_product_edges(fgg, rule.rhs.nodes(), rule.rhs.edges(), rule.rhs.ext, inputs, semiring=semiring))
            elif len(edges) == 1:
                [edge] = edges
                J0.add_single((n, edge.label), sum_product_edges(fgg, rule.rhs.nodes(), set(rule.rhs.edges()) - {edge}, rule.rhs.ext + edge.nodes, inputs, semiring=semiring))
            else:
                rhs = ' '.join(e.label.name for e in edges)
                raise ValueError(f'FGG is not linearly recursive ({rule.lhs.name} -> {rhs})')

    return multi_solve(J0, F0)


class SumProduct(torch.autograd.Function):
    """Compute the sum-product of a subset of the nonterminals of an FGG.

    The interface is slightly awkward because PyTorch does not
    autodifferentiate with respect to arguments that are not Tensors.

    - fgg: The FGG
    - opts: A dict of options (see documentation for sum_product).
    - in_labels: The (terminal or nonterminal) EdgeLabels whose
      sum-products are already computed.
    - out_labels: The nonterminal EdgeLabels whose sum-products to compute.
    - in_values: A sequence of Tensors such that in_values[i] is the
      sum-product of in_labels[i].

    Returns: A sequence of Tensors out_values such that out_values[i]
    is the sum-product of out_labels[i].
    """
    
    @staticmethod
    def forward(ctx, fgg: FGG, opts: Dict, in_labels: Sequence[EdgeLabel], out_labels: Sequence[EdgeLabel], *in_values: Tensor) -> Tuple[Tensor, ...]: # type: ignore
        ctx.fgg = fgg
        method, semiring = opts['method'], opts['semiring']
        ctx.opts = opts
        ctx.in_labels = in_labels
        ctx.out_labels = out_labels
        ctx.save_for_backward(*in_values)

        inputs: MultiTensor = dict(zip(in_labels, in_values)) # type: ignore

        if method == 'linear':
            out = linear(fgg, inputs, out_labels, semiring)
        else:
            x0 = MultiTensor(FGGMultiShape(fgg, out_labels), semiring)
            if method == 'fixed-point':
                fixed_point(lambda x: F(fgg, x, inputs, semiring),
                            x0, tol=opts['tol'], kmax=opts['kmax'])
            elif method == 'newton':
                newton(lambda x: F(fgg, x, inputs, semiring),
                       lambda x: J(fgg, x, inputs, semiring),
                       x0, tol=opts['tol'], kmax=opts['kmax'])
            elif method == 'one-step':
                x0.copy_(F(fgg, x0, inputs, semiring))
            else:
                raise ValueError('unsupported method for computing sum-product')
            out = x0

        ctx.out_values = out
        return tuple(out[nt] if nt in out else semiring.zeros(fgg.shape(nt))
                     for nt in out_labels)

    @staticmethod
    def backward(ctx, *grad_out):
        semiring = ctx.opts['semiring']
        # gradients are always computed in the real semiring
        real_semiring = RealSemiring(dtype=semiring.dtype, device=semiring.device)

        # Construct and solve linear system of equations
        inputs = dict(zip(ctx.in_labels, ctx.saved_tensors))
        f = dict(zip(ctx.out_labels, grad_out))
            
        jf_inputs = MultiTensor((FGGMultiShape(ctx.fgg, ctx.out_labels),
                                 FGGMultiShape(ctx.fgg, ctx.in_labels)),
                                real_semiring)
        if isinstance(semiring, RealSemiring):
            jf = J(ctx.fgg, ctx.out_values, inputs, semiring, jf_inputs)
        elif isinstance(semiring, LogSemiring):
            jf = J_log(ctx.fgg, ctx.out_values, inputs, semiring, jf_inputs)
        else:
            raise ValueError(f'invalid semiring: {semiring}')

        grad_nt = multi_solve(jf, f, transpose=True)
                    
        # Compute gradients of inputs
        grad_t = multi_mv(jf_inputs, grad_nt, transpose=True)
        grad_in = tuple(grad_t[el] for el in ctx.in_labels)
        
        return (None, None, None, None) + grad_in


def sum_products(fgg: FGG, **opts) -> Dict[EdgeLabel, Tensor]:
    opts.setdefault('method',   'fixed-point')
    opts.setdefault('semiring', RealSemiring())
    opts.setdefault('tol',      1e-5) # with float32, 1e-6 can fail
    opts.setdefault('kmax',     1000) # for fixed-point, 100 is too low
    if isinstance(opts['semiring'], BoolSemiring):
        opts['tol'] = 0

    all = {t:cast(FiniteFactor, fgg.factors[t.name]).weights for t in fgg.terminals()}
    for comp in scc(nonterminal_graph(fgg)):

        inputs = {}
        max_rhs = 0
        for x in comp:
            for r in fgg.rules(x):
                n = 0
                for e in r.rhs.edges():
                    if e.label in comp:
                        n += 1
                    else:
                        inputs[e.label] = all[e.label]
                max_rhs = max(max_rhs, n)

        comp_opts = dict(opts)
        if len(comp) == 1 and max_rhs == 0:
            # SCC has a single, non-looping nonterminal
            comp_opts['method'] = 'one-step'
        elif max_rhs == 1 and opts['method'] == 'newton':
            # SCC is linearly recursive
            comp_opts['method'] = 'linear'

        comp_labels = list(comp)
        comp_values = SumProduct.apply(fgg, comp_opts, inputs.keys(), comp_labels, *inputs.values())
        all.update(zip(comp_labels, comp_values))
    return all
        
def sum_product(fgg: FGG, **opts) -> Tensor:
    """Compute the sum-product of an FGG.
    
    - fgg: The FGG to compute the sum-product of.
    - method: What method to use ('linear', 'fixed-point', 'newton').
    - tol: Iterative algorithms terminate when the L∞ distance between consecutive iterates is below tol.
    - kmax: Number of iterations after which iterative algorithms give up.
    """
    all = sum_products(fgg, **opts)
    if fgg.start is None:
        raise ValueError("FGG must have a start symbol")
    return all[fgg.start]
