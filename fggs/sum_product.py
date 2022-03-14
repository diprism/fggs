__all__ = ['sum_product']

from fggs.fggs import FGG, HRG, HRGRule, Interpretation, EdgeLabel, Edge, Node
from fggs.factors import CategoricalFactor
from fggs.semirings import *
from fggs.multi import *

from typing import Callable, Dict, Mapping, Sequence, Iterable, Tuple, List, Set, Union, Optional
import warnings

import torch
from torch import Tensor
import torch_semiring_einsum

Function = Callable[[MultiTensor], MultiTensor]

def _formatwarning(message, category, filename=None, lineno=None, file=None, line=None):
    return '%s:%s: %s: %s' % (filename, lineno, category.__name__, message)
warnings.formatwarning = _formatwarning # type: ignore

def tensordot(a, b, n):
    # https://github.com/pytorch/pytorch/issues/61096 (PyTorch 1.9.0)
    return torch.tensordot(a, b, n) if n > 0 \
        else a.reshape(a.size() + (1,) * b.dim()) * b

def scc(g: HRG) -> List[HRG]:
    """Decompose an HRG into a its strongly-connected components using Tarjan's algorithm.

    Returns a list of sets of nonterminal EdgeLabels. The list is in
    topological order: there is no rule with a lhs in an earlier
    component and an rhs nonterminal in a later component.

    Robert Tarjan. Depth-first search and linear graph
    algorithms. SIAM J. Comput., 1(2),
    146-160. https://doi.org/10.1137/0201010

    Based on pseudocode from https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm

    """
    
    index = 0
    indexof = {}    # order of nonterminals in DFS traversal
    lowlink = {}    # lowlink[v] = min(indexof[w] | w is v or a descendant of v)
    stack = []      # path from start nonterminal to current nonterminal
    onstack = set() # = set(stack)
    comps = []

    def visit(v):
        nonlocal index
        indexof[v] = lowlink[v] = index
        index += 1
        stack.append(v)
        onstack.add(v)

        for r in g.rules(v):
            nts = set(e.label for e in r.rhs.edges() if e.label.is_nonterminal)
            for w in nts:
                if w not in indexof:
                    visit(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in onstack:
                    lowlink[v] = min(lowlink[v], indexof[w])

        if lowlink[v] == indexof[v]:
            comp = set()
            while v not in comp:
                w = stack.pop()
                onstack.remove(w)
                comp.add(w)
            comps.append(comp)
    
    for v in g.nonterminals():
        if v not in indexof:
            visit(v)

    return comps


class FGGMultiShape(MultiShape):
    """A virtual Multishape for the nonterminals or terminals of an FGG."""
    def __init__(self, fgg, els):
        self.fgg = fgg
        self.els = list(els)
    def __getitem__(self, x):
        return torch.Size(self.fgg.interp.shape(x))
    def __iter__(self):
        return iter(self.els)
    def __len__(self):
        return len(self.els)

    
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
    k = 0
    x1 = MultiTensor(x0.shapes, x0.semiring)
    for k in range(kmax):
        F0 = F(x0)
        JF = J(x0)
        dX = multi_solve(JF, F0 - x0)
        x1.copy_(x0 + dX)
        if x0.allclose(x1, tol): break
        x0.copy_(x1)
    if k > kmax:
        warnings.warn('maximum iteration exceeded; convergence not guaranteed')


def F(fgg: FGG, x: MultiTensor, inputs: MultiTensor, semiring: Semiring) -> MultiTensor:
    hrg, interp = fgg.grammar, fgg.interp
    Fx = MultiTensor(x.shapes, x.semiring)
    for n in hrg.nonterminals():
        for rule in hrg.rules(n):
            tau_rule = sum_product_edges(interp, rule.rhs.nodes(), rule.rhs.edges(), rule.rhs.ext, x, inputs, semiring=semiring)
            Fx.add_single(n, tau_rule)
    return Fx


def J(fgg: FGG, x: MultiTensor, inputs: MultiTensor, semiring: Semiring,
      J_terminals: Optional[MultiTensor] = None) -> MultiTensor:
    """The Jacobian of F."""
    hrg, interp = fgg.grammar, fgg.interp
    Jx = MultiTensor(x.shapes+x.shapes, semiring)
    for n in hrg.nonterminals():
        for rule in hrg.rules(n):
            for edge in rule.rhs.edges():
                if edge.label.is_terminal and J_terminals is None: continue
                ext = rule.rhs.ext + edge.nodes
                edges = set(rule.rhs.edges()) - {edge}
                tau_edge = sum_product_edges(interp, rule.rhs.nodes(), edges, ext, x, inputs, semiring=semiring)
                if edge.label.is_terminal:
                    assert J_terminals is not None
                    J_terminals.add_single((n, edge.label), tau_edge)
                else:
                    Jx.add_single((n, edge.label), tau_edge)
    return Jx


def J_log(fgg: FGG, x: MultiTensor, inputs: MultiTensor, semiring: Semiring,
          J_terminals: Optional[MultiTensor] = None) -> MultiTensor:
    """The Jacobian of F(semiring=LogSemiring), computed in the real semiring."""
    hrg, interp = fgg.grammar, fgg.interp
    Jx = MultiTensor(x.shapes+x.shapes, semiring=RealSemiring(dtype=semiring.dtype, device=semiring.device))
    for n in hrg.nonterminals():
        rules = list(hrg.rules(n))
        tau_rules: Union[List[Tensor], Tensor]
        tau_rules = []
        for rule in rules:
            tau_rule = sum_product_edges(interp, rule.rhs.nodes(), rule.rhs.edges(), rule.rhs.ext, x, inputs, semiring=semiring)
            tau_rules.append(tau_rule)
        tau_rules = torch.stack(tau_rules, dim=0)
        tau_rules = torch.log_softmax(tau_rules, dim=0).nan_to_num()
        for rule, tau_rule in zip(rules, tau_rules):
            for edge in rule.rhs.edges():
                if edge.label.is_terminal and J_terminals is None: continue
                ext = rule.rhs.ext + edge.nodes
                tau_edge = sum_product_edges(interp, rule.rhs.nodes(), rule.rhs.edges(), ext, x, inputs, semiring=semiring)
                tau_edge_size = tau_edge.size()
                tau_edge = tau_edge.reshape(tau_rule.size() + (-1,))
                tau_edge = torch.log_softmax(tau_edge, dim=-1).nan_to_num()
                tau_edge += tau_rule.unsqueeze(-1)
                tau_edge = tau_edge.reshape(tau_edge_size)
                if edge.label.is_terminal:
                    assert J_terminals is not None
                    J_terminals.add_single((n, edge.label), torch.exp(tau_edge))
                else:
                    Jx.add_single((n, edge.label), torch.exp(tau_edge))
    return Jx


def sum_product_edges(interp: Interpretation, nodes: Iterable[Node], edges: Iterable[Edge], ext: Sequence[Node], *inputses: MultiTensor, semiring: Semiring) -> Tensor:
    """Compute the sum-product of a set of edges.

    Parameters:
    - interp
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
            nsize = interp.domains[n.label].size()
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
            if edge.label.is_nonterminal:
                # One argument to einsum will be the zero tensor, so just return zero
                return semiring.zeros(interp.shape(ext))
            else:
                raise TypeError(f'cannot compute sum-product of FGG with factor {interp.factors[edge.label]}')

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
        vshape = [interp.domains[n.label].size() if n in connected else 1 for n in ext]
        out = out.view(*vshape).repeat(*interp.shape(ext))

    # Multiply in any disconnected internal nodes.
    mul = 1
    for n in nodes:
        if n not in connected and n not in ext:
            mul *= interp.domains[n.label].size()
    if mul > 1:
        out = semiring.mul(out, semiring.from_int(mul))

    return out


def linear(fgg: FGG, inputs: MultiTensor, semiring: Semiring) -> MultiTensor:
    """Compute the sum-product of the nonterminals of `fgg`, which is
    linearly recursive given that each nonterminal `x` in `inputs` is
    treated as a terminal with weight `inputs[x]`.
    """
    hrg, interp = fgg.grammar, fgg.interp
    shapes = FGGMultiShape(fgg, hrg.nonterminals())

    # Check linearity and compute F(0) and J(0)
    F0 = MultiTensor(shapes, semiring)
    J0 = MultiTensor((shapes, shapes), semiring)
    for n in hrg.nonterminals():
        if n in inputs:
            continue
        for rule in hrg.rules(n):
            edges = [e for e in rule.rhs.edges() if e.label.is_nonterminal and e.label not in inputs]
            if len(edges) == 0:
                F0.add_single(n, sum_product_edges(interp, rule.rhs.nodes(), rule.rhs.edges(), rule.rhs.ext, inputs, semiring=semiring))
            elif len(edges) == 1:
                [edge] = edges
                J0.add_single((n, edge.label), sum_product_edges(interp, rule.rhs.nodes(), set(rule.rhs.edges()) - {edge}, rule.rhs.ext + edge.nodes, inputs, semiring=semiring))
            else:
                raise ValueError('FGG is not linearly recursive')

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
        opts.setdefault('method',   'fixed-point')
        opts.setdefault('semiring', RealSemiring())
        opts.setdefault('tol',      1e-6)
        opts.setdefault('kmax',     1000)
        method, semiring = opts['method'], opts['semiring']
        if isinstance(semiring, BoolSemiring):
            opts['tol'] = 0
        
        ctx.opts = opts
        ctx.in_labels = in_labels
        ctx.out_labels = out_labels
        ctx.save_for_backward(*in_values)

        inputs: MultiTensor = dict(zip(in_labels, in_values)) # type: ignore

        if method == 'linear':
            out = linear(fgg, inputs, semiring)
        else:
            x0 = MultiTensor(FGGMultiShape(fgg, fgg.grammar.nonterminals()), semiring)
            if method == 'fixed-point':
                fixed_point(lambda x: F(fgg, x, inputs, semiring),
                            x0, tol=opts['tol'], kmax=opts['kmax'])
            elif method == 'newton':
                newton(lambda x: F(fgg, x, inputs, semiring),
                       lambda x: J(fgg, x, inputs, semiring),
                       x0, tol=opts['tol'], kmax=opts['kmax'])
            else:
                raise ValueError('unsupported method for computing sum-product')
            out = x0

        ctx.out_values = out
        return tuple(out[nt] if nt in out else semiring.zeros(fgg.interp.shape(nt))
                     for nt in out_labels)

    @staticmethod
    def backward(ctx, *grad_out):
        hrg, interp = ctx.fgg.grammar, ctx.fgg.interp
        shapes = FGGMultiShape(ctx.fgg, hrg.nonterminals())
        semiring = ctx.opts['semiring']
        # gradients are always computed in the real semiring
        real_semiring = RealSemiring(dtype=semiring.dtype, device=semiring.device)

        # Construct and solve linear system of equations
        inputs = dict(zip(ctx.in_labels, ctx.saved_tensors))
        f = dict(zip(ctx.out_labels, grad_out))
            
        jf_terminals = MultiTensor((shapes, FGGMultiShape(ctx.fgg, hrg.terminals())), real_semiring)
        if isinstance(semiring, RealSemiring):
            jf = J(ctx.fgg, ctx.out_values, inputs, semiring, jf_terminals)
        elif isinstance(semiring, LogSemiring):
            jf = J_log(ctx.fgg, ctx.out_values, inputs, semiring, jf_terminals)
        else:
            raise ValueError(f'invalid semiring: {semiring}')

        grad_nt = multi_solve(jf, f, transpose=True)

        # Change infs to very large numbers
        for x in grad_nt:
            torch.nan_to_num(grad_nt[x], out=grad_nt[x])
                    
        # Compute gradients of factors
        grad_t = multi_mv(jf_terminals, grad_nt, transpose=True)
        grad_in = tuple(grad_t[el] if el.is_terminal else grad_nt[el] for el in ctx.in_labels)
        
        return (None, None, None, None) + grad_in


def sum_product(fgg: FGG, **opts) -> Tensor:
    """Compute the sum-product of an FGG.
    
    - fgg: The FGG to compute the sum-product of.
    - method: What method to use ('linear', 'fixed-point', 'newton').
    - tol: Iterative algorithms terminate when the L∞ distance between consecutive iterates is below tol.
    - kmax: Number of iterations after which iterative algorithms give up.
    """

    hrg, interp = fgg.grammar, fgg.interp
    in_labels = list(hrg.terminals())
    in_values = [interp.factors[t].weights for t in in_labels]
    out_labels = list(hrg.nonterminals())
    out = SumProduct.apply(fgg, opts, in_labels, out_labels, *in_values)
    return out[out_labels.index(fgg.grammar.start_symbol)]
