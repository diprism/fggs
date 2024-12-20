from __future__ import annotations
__all__ = ['sum_product', 'sum_products']
from fggs.fggs import FGG, HRG, HRGRule, EdgeLabel, Edge, Node
from fggs.factors import FiniteFactor
from fggs.semirings import *
from fggs.multi import *
from fggs.utils import scc, nonterminal_graph
from itertools import chain
from math import inf
from sys import stderr

from typing import Callable, Dict, Mapping, Sequence, Iterable, Tuple, List, Set, Union, Optional, cast
from time import perf_counter_ns
import warnings

import torch
from torch import Tensor
from fggs.typing import TensorLikeT
from fggs.indices import Nonphysical, PatternedTensor, einsum, stack
from typing import cast
Function = Callable[[MultiTensor], MultiTensor]

def _formatwarning(message, category, filename=None, lineno=None, file=None, line=None):
    return '%s:%s: %s: %s' % (filename, lineno, category.__name__, message)
warnings.formatwarning = _formatwarning # type: ignore


class FGGMultiShape(MultiShape):
    """A virtual MultiShape for the nonterminals or terminals of an FGG."""
    def __init__(self, fgg, els):
        self.fgg = fgg
        self.els = list(els)
        self.elset = set(self.els)
    def __getitem__(self, x):
        if x in self.elset:
            return torch.Size(self.fgg.shape(x))
        else:
            raise KeyError()
    def __contains__(self, x):
        return x in self.elset
    def __iter__(self):
        return iter(self.els)
    def __len__(self):
        return len(self.els)
    def __str__(self):
        return str(self.els)
    def __repr__(self):
        return repr(self.els)


def make_timer(name: str) -> Callable[[], None]:
    n = 0
    t = perf_counter_ns()
    def timer():
        nonlocal n, t
        now = perf_counter_ns()
        n += 1
        ns = now - t
        print(f'{name} iteration {n} took {ns//1000000000:,}.{ns%1000000000:011,} sec', file=stderr)
        t = now
    return timer

    
def fixed_point(F: Function, x0: MultiTensor, *, tol: float, kmax: int) -> None:
    """Fixed-point iteration method for solving x = F(x)."""
    #timer = make_timer('fixed_point')
    k, x1 = 0, F(x0)
    #timer()
    while not x0.shouldStop(x1, tol) and k <= kmax:
        x0.copy_(x1)
        x1.copy_(F(x1))
        k += 1
        #timer()
    if k > kmax:
        warnings.warn('maximum iteration exceeded; convergence not guaranteed')


def newton(F: Function, J: Function, x0: MultiTensor, *, tol: float, kmax: int) -> None:
    """Newton's method for solving x = F(x) in a commutative semiring.

    Javier Esparza, Stefan Kiefer, and Michael Luttenberger. On fixed
    point equations over commutative semirings. In Proc. STACS, 2007."""
    semiring = x0.semiring
    x1 = MultiTensor(x0.shapes, x0.semiring)
    #timer = make_timer('newton')
    for k in range(kmax):
        # The inequality x0 <= F(x0) <= x0+dX is theoretically guaranteed
        # (Etessami and Yannakakis), but due to rounding error it may fail to
        # hold in practice, preventing convergence. So we use maximum_ twice
        # below to force it to hold (cf. Nederhof and Satta, who suggest a
        # similar trick). The first use of maximum_ seems especially helpful.
        F0 = F(x0).maximum_(x0)
        stop = F0.shouldStop(x0, tol)
        JF = J(x0)
        dX = multi_solve(JF, F0 - x0)
        #^ For the derivative blowup issue, it's only JF that's bigger than it needs to be, so dX above could be back to an ordinary dense tensor
        x0 += dX
        x0.maximum_(F0)
        #timer()
        if stop: break

    if k > kmax:
        warnings.warn('maximum iteration exceeded; convergence not guaranteed')


def F(fgg: FGG, x: MultiTensor, inputs: MultiTensor, semiring: Semiring) -> MultiTensor:
    Fx = MultiTensor(x.shapes, x.semiring)
    for n in x.shapes[0]:
        for rule in fgg.rules(n):
            tau_rule = sum_product_edges(fgg, rule.rhs.nodes(), rule.rhs.edges(), rule.rhs.ext, x, inputs, semiring=semiring)
            if tau_rule is not None: Fx.add_single(n, tau_rule)
    return Fx


def print_duplicate(loc: str, xs: Sequence[Node], ys: Sequence[Node]) -> bool:
    return False # Comment out this line for debugging messages
    duplicate = frozenset(xs).intersection(ys)
    if duplicate:
        print('Duplicate in', loc, 'between', file=stderr)
        for n in xs: print('  * ' if n in duplicate else '  - ', str(n), file=stderr)
        print('and', file=stderr)
        for n in ys: print('  * ' if n in duplicate else '  - ', str(n), file=stderr)
        return True
    else:
        return False


def J(fgg: FGG, x: MultiTensor, inputs: MultiTensor, semiring: Semiring,
      J_inputs: Optional[MultiTensor] = None) -> MultiTensor:
    """The Jacobian of F."""
    Jx = MultiTensor(x.shapes+x.shapes, semiring)
    for n in x.shapes[0]:
        for rule in fgg.rules(n):
            for edge in rule.rhs.edges():
                if edge.label not in Jx.shapes[1] and J_inputs is None: continue
                duplicate = print_duplicate('J', rule.rhs.ext, edge.nodes)
                ext = rule.rhs.ext + edge.nodes
                edges = set(rule.rhs.edges()) - {edge}
                tau_edge = sum_product_edges(fgg, rule.rhs.nodes(), edges, ext, x, inputs, semiring=semiring)
                if tau_edge is not None:
                    if duplicate:
                        print('sum_product_edges produced', tau_edge.physical.size(), 'for', tau_edge.size(), file=stderr)
                    if edge.label in Jx.shapes[1]:
                        Jx.add_single((n, edge.label), tau_edge)
                    elif J_inputs is not None and edge.label in J_inputs.shapes[1]:
                        J_inputs.add_single((n, edge.label), tau_edge)
                    else:
                        assert False
    return Jx

def J_precompute_products(fgg: FGG, x: MultiTensor, inputs: MultiTensor, semiring: Semiring,
      J_inputs: Optional[MultiTensor] = None) -> MultiTensor:
    """The Jacobian of F. New version that takes out the double loop over edges"""
    Jx = MultiTensor(x.shapes+x.shapes, semiring)
    for n in x.shapes[0]:
        for rule in fgg.rules(n):
            edges = list(rule.rhs.edges())
            if len(edges) == 1:
                suffix_products = [sum_product_edges(fgg, rule.rhs.nodes(), set(), rule.rhs.ext + edges[0].nodes, x, inputs, semiring=semiring)]
            else:
                prefix_products, prefix_output_nodes = compute_products(fgg, (x, inputs), edges, rule, semiring)
                suffix_products, suffix_output_nodes = compute_products(fgg, (x, inputs), list(reversed(edges)), rule, semiring)
                suffix_products.reverse()
                suffix_output_nodes.reverse()
            for i, edge in enumerate(edges):
                if i == 0:
                    product = suffix_products[0]
                elif i == len(edges) - 1:
                    product = prefix_products[-1]
                else:
                    if prefix_products[i - 1] is None or suffix_products[i] is None: continue
                    #casts from optional to satisfy mypy
                    prefix_product: PatternedTensor = cast(PatternedTensor, prefix_products[i - 1])
                    suffix_product: PatternedTensor = cast(PatternedTensor, suffix_products[i])
                    prefix_output_node: List[Node]  = cast(List[Node], prefix_output_nodes[i - 1])
                    suffix_output_node: List[Node]  = cast(List[Node], suffix_output_nodes[i])
                    product = einsum([prefix_product, suffix_product],
                                     [prefix_output_node, suffix_output_node],
                                     [n for n in rule.rhs.ext + edge.nodes if n in prefix_output_node or n in suffix_output_node], 
                                     semiring)
                if product is not None:
                    # add back removed disconnected external nodes
                    ext = list(rule.rhs.ext) + list(edge.nodes)
                    if product.ndim < len(ext):
                        connected = set()
                        for e in edges:
                            if i != 0 and i != len(edges) -1 and e is edge: continue
                            connected.update(e.nodes)
                        eshape = fgg.shape(ext)
                        vshape = [s if n in connected else 1 for n, s in zip(ext, eshape)]
                        product = product.view(*vshape).expand(*eshape)
                    if edge.label in Jx.shapes[1]:
                        Jx.add_single((n, edge.label), product)    
                    elif J_inputs is not None and edge.label in J_inputs.shapes[1]:
                        J_inputs.add_single((n, edge.label), product)
    return Jx

def compute_products(fgg: FGG, inputses: Iterable[MultiTensor], edges: List[Edge], rule: HRGRule, semiring: Semiring) -> Tuple[List[Optional[PatternedTensor]], List[Optional[List[Node]]]]:
    future_nodes_lookup = []
    future_nodes = []
    seen_nodes = set()
    for edge in reversed(edges[1:]):
        for node in edge.nodes:
            if node not in seen_nodes:
                future_nodes.append(node)
                seen_nodes.add(node)

        future_nodes_lookup.append(future_nodes.copy())
    future_nodes_lookup.reverse()

    products: List[Optional[PatternedTensor]] = []
    output_nodes: List[Optional[List[Node]]] = [] 
    previous_weight: PatternedTensor = PatternedTensor.eye(1, semiring=semiring)
    previous_output_nodes: List[Node] = []
    for i, edge in enumerate(edges[:-1]):
        weight = get_weight(edge, inputses)
        if weight is None:
            products.extend( [None] * (len(edges) - 1 - i))
            output_nodes.extend( [None] * (len(edges) - 1 - i))
            break
        out, out_nodes = multiply_next_edge(fgg, previous_weight, previous_output_nodes, weight, list(edge.nodes), 
                                            list(rule.rhs.ext) + future_nodes_lookup[i], list(rule.rhs.nodes()), semiring)
        products.append(out)
        output_nodes.append(out_nodes)
        previous_weight, previous_output_nodes = out, out_nodes
        
    return products, output_nodes

def multiply_next_edge(fgg: FGG, previous_weight: PatternedTensor, previous_nodes: List[Node], 
                       current_weight: PatternedTensor, current_nodes: List[Node], 
                       ext: Iterable[Node], rule_rhs_nodes: List[Node], semiring: Semiring) -> Tuple[PatternedTensor, List[Node]]:
    
    indexing: List[Sequence[Node]] = [previous_nodes, current_nodes]
    tensors: List[PatternedTensor] = [previous_weight, current_weight]
    connected: Set[Node] = set(previous_nodes + current_nodes)

    ext, duplicate = rename_duplicate_nodes(fgg, ext, tensors, indexing, connected, semiring)
    output_nodes = [n for n in ext if n in connected]
    
    assert(all(tensor.physical.dtype == semiring.dtype for tensor in tensors))
    out = einsum(tensors, indexing, output_nodes, semiring)
    if duplicate:
        print('einsum produced', out.physical.size(), 'for', out.size(), file=stderr)

    out = multiply_in_disconnected_internals(out, rule_rhs_nodes, connected, ext, semiring, fgg)
    return out, output_nodes

def log_softmax(a: TensorLikeT, dim: int) -> TensorLikeT:
    # If a has infinite elements, log_softmax would return all nans.
    # In this case, make all the nonzero elements 1 and the zero elements 0.
    return a.gt(-inf).log().to(dtype=a.dtype).where(a.eq(inf).any(dim, keepdim=True),
                                                    a.log_softmax(dim))
    
def J_log(fgg: FGG, x: MultiTensor, inputs: MultiTensor, semiring: Semiring,
          J_inputs: Optional[MultiTensor] = None) -> MultiTensor:
    """The Jacobian of F(semiring=LogSemiring), computed in the real semiring."""
    Jx = MultiTensor(x.shapes+x.shapes, semiring=RealSemiring(dtype=semiring.dtype, device=semiring.device))
    for n in x.shapes[0]:
        rules = tuple(fgg.rules(n))
        tau_rule_pairs: List[Tuple[HRGRule, PatternedTensor]] = \
            [(rule, tau_rule)
             for rule in rules
             for tau_rule in (sum_product_edges(fgg, rule.rhs.nodes(), rule.rhs.edges(),
                                                rule.rhs.ext, x, inputs, semiring=semiring),)
             if tau_rule is not None]
        if len(tau_rule_pairs) == 0: continue
        rules, tau_rules = zip(*tau_rule_pairs)
        tau_rules_stacked : PatternedTensor = stack(tau_rules, dim=0)
        tau_rules_stacked = log_softmax(tau_rules_stacked, dim=0)
        for rule, tau_rule in zip(rules, tau_rules_stacked):
            for edge in rule.rhs.edges():
                if edge.label not in Jx.shapes[1] and J_inputs is None: continue
                duplicate = print_duplicate('J_log', rule.rhs.ext, edge.nodes)
                ext = rule.rhs.ext + edge.nodes
                tau_edge = sum_product_edges(fgg, rule.rhs.nodes(), rule.rhs.edges(), ext, x, inputs, semiring=semiring)
                if tau_edge is not None:
                    if duplicate:
                        print('sum_product_edges produced', tau_edge.physical.size(), 'for', tau_edge.size(), file=stderr)
                    tau_edge_size = tau_edge.size()
                    tau_edge = tau_edge.reshape(tau_rule.size() + (-1,))
                    tau_edge = log_softmax(tau_edge, dim=-1)
                    tau_edge = tau_edge.add(tau_rule.unsqueeze(-1))
                    tau_edge = tau_edge.reshape(tau_edge_size)
                    if edge.label in Jx.shapes[1]:
                        Jx.add_single((n, edge.label), tau_edge.exp())
                    elif J_inputs is not None and edge.label in J_inputs.shapes[1]:
                        J_inputs.add_single((n, edge.label), tau_edge.exp())
                    else:
                        assert False
    return Jx


def multiply_in_disconnected_internals(out: PatternedTensor, nodes: Iterable[Node], connected: Set[Node], ext: Iterable[Node], semiring: Semiring, fgg: FGG) -> PatternedTensor:
    """Multiply in any disconnected internal nodes"""
    multiplier = 1
    for n in nodes:
        if n not in connected and n not in ext:
            multiplier *= fgg.domains[n.label.name].size()
    if multiplier != 1:
        return semiring.mul(out, PatternedTensor.from_int(multiplier, semiring))
    return out

def rename_duplicate_nodes(fgg: FGG, ext: Iterable[Node], tensors: List[PatternedTensor], indexing: List[Sequence[Node]], connected: Set[Node], semiring: Semiring):
    """ Derivatives can sometimes produce duplicate external nodes.
        Rename them apart and add identity factors between them."""
    ext_orig = ext
    ext = []
    debug_duplicate = False
    for n in ext_orig:
        if n in ext:
            ncopy = Node(n.label)
            ext.append(ncopy)
            connected.update([n, ncopy])
            indexing.append([n, ncopy])
            nsize = fgg.domains[n.label.name].size()
            tensors.append(PatternedTensor.eye(nsize,semiring))
            #debug_duplicate = True # Uncomment this line for debugging messages
        else:
            ext.append(n)

    return ext, debug_duplicate

def get_weight(edge, inputses):
    for inputs in inputses:
        if edge.label in inputs:
            return inputs[edge.label]
    return None

def sum_product_edges(fgg: FGG, nodes: Iterable[Node], edges: Iterable[Edge], ext: Sequence[Node], *inputses: MultiTensor, semiring: Semiring) -> Optional[PatternedTensor]:
    """Compute the sum-product of a set of edges.

    Parameters:
    - fgg
    - nodes: all the nodes, even disconnected ones
    - edges: the edges whose factors are multiplied together
    - ext: the nodes whose values are not summed over
    - inputses: dicts of sum-products of nonterminals that have
      already been computed. Later elements of inputses override
      earlier elements.

    Return: the tensor of sum-products, or None if zero
    """
    connected: Set[Node] = set()
    indexing: List[Sequence[Node]] = []
    tensors: List[PatternedTensor] = []
    ext, duplicate = rename_duplicate_nodes(fgg, ext, tensors, indexing, connected, semiring)

    for edge in edges:
        connected.update(edge.nodes)
        indexing.append(edge.nodes)
        weight = get_weight(edge, inputses)
        if weight is not None:
            tensors.append(weight)
        else:
            return None  #One argument to einsum will be the zero tensor, so just return zero

    # If an external node has no edges, einsum will complain, so remove it.
    outputs = [node for node in ext if node in connected]

    assert(all(tensor.physical.dtype == semiring.dtype for tensor in tensors))
    out = einsum(tensors, indexing, outputs, semiring)
    if duplicate:
        print('einsum produced', out.physical.size(), 'for', out.size(), file=stderr)

    # Restore any external nodes that were removed.
    if out.ndim < len(ext):
        eshape = fgg.shape(ext)
        vshape = [s if n in connected else 1 for n, s in zip(ext, eshape)]
        out = out.view(*vshape).expand(*eshape)

    out = multiply_in_disconnected_internals(out, nodes, connected, ext, semiring, fgg)

    assert(out.physical.dtype == semiring.dtype)
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
                t = sum_product_edges(fgg, rule.rhs.nodes(), rule.rhs.edges(),
                                      rule.rhs.ext, inputs, semiring=semiring)
                if t is not None:
                    F0.add_single(n, t)
            elif len(edges) == 1:
                [edge] = edges
                duplicate = print_duplicate('linear', rule.rhs.ext, edge.nodes)
                t = sum_product_edges(fgg, rule.rhs.nodes(), set(rule.rhs.edges()) - {edge},
                                      rule.rhs.ext + edge.nodes, inputs, semiring=semiring)
                if t is not None:
                    if duplicate:
                        print('sum_product_edges produced', t.physical.size(), 'for', t.size(), file=stderr)
                    J0.add_single((n, edge.label), t)
            else:
                rhs = ' '.join(e.label.name for e in edges)
                raise ValueError(f'FGG is not linearly recursive ({rule.lhs.name} -> {rhs})')

    return multi_solve(J0, F0)


class SumProduct(torch.autograd.Function):
    """Compute the sum-product of a subset of the nonterminals of an FGG.

    The interface is awkward because PyTorch does not autodifferentiate
    with respect to arguments and outputs that are not Tensors.

    - fgg: The FGG
    - opts: A dict of options (see documentation for sum_product).
    - in_labels: A sequence of sum-products already computed.
      For each (terminal or nonterminal) EdgeLabel whose sum-products
      are already computed, the element in_labels[i] is a pair
      (el[i], in_np[i]) where
      - el[i] is the EdgeLabel and
      - in_np[i] is the nonphysical part of the PatternedTensor
        that is the sum-product of el[i].
        A nonphysical part of a PatternedTensor pt is a Nonphysical object
        whose reincarnate() method produces pt when passed the Tensor
        pt.physical.
    - out_labels: The nonterminal EdgeLabels whose sum-products to compute.
    - in_values: A sequence of Tensors such that the PatternedTensor
      in_np[i].reincarnate(in_values[i]) is the sum-product of in_labels[i].

    Returns: A sequence (out_np, *out_values), where
    - out_np consists of len(out_labels) nonphysical parts and
    - out_values consists of len(out_labels) Tensors,
    such that out_np[i].reincarnate(out_values[i]) is the sum-product of
    out_labels[i].
    """
    
    @staticmethod
    def forward(ctx, # type: ignore
                fgg: FGG,
                opts: Dict,
                in_labels: Sequence[Tuple[EdgeLabel, Nonphysical]],
                out_labels: Sequence[EdgeLabel],
                *in_values: Tensor) -> Tuple[Union[
            Tuple[Optional[Nonphysical], ...], # first tuple component (same length as out_labels)
            Optional[Tensor]                   # rest tuple components (same length as out_labels)
        ], ...]:
        ctx.fgg = fgg
        method, semiring, j_precompute = opts['method'], opts['semiring'], opts.get('j_precompute', False)
        ctx.opts = opts
        ctx.in_labels = in_labels
        ctx.out_labels = out_labels
        ctx.save_for_backward(*in_values)
        

        inputs: MultiTensor = {label: nonphysical.reincarnate(physical) for (label, nonphysical), physical in zip(in_labels, in_values)} # type: ignore
        if method == 'linear':
            out = linear(fgg, inputs, out_labels, semiring)
        else:
            x0 = MultiTensor(FGGMultiShape(fgg, out_labels), semiring)
            if method == 'fixed-point':
                fixed_point(lambda x: F(fgg, x, inputs, semiring),
                            x0, tol=opts['tol'], kmax=opts['kmax'])
            elif method == 'newton':
                newton(lambda x: F(fgg, x, inputs, semiring),
                       lambda x: J_precompute_products(fgg, x, inputs, semiring)
                       if j_precompute else J(fgg, x, inputs, semiring),
                       x0, tol=opts['tol'], kmax=opts['kmax'])
            elif method == 'one-step':
                x0.copy_(F(fgg, x0, inputs, semiring))
            else:
                raise ValueError('unsupported method for computing sum-product')
            out = x0

        ctx.out_values = out
        return tuple(chain((tuple(out[nt].nonphysical() if nt in out else None
                                  for nt in out_labels),),
                           (out[nt].physical if nt in out else None
                            for nt in out_labels)))

    @staticmethod
    def backward(ctx, grad_nonphysicals, *grad_out):
        semiring = ctx.opts['semiring']
        # gradients are always computed in the real semiring
        real_semiring = RealSemiring(dtype=semiring.dtype, device=semiring.device)

        # Construct and solve linear system of equations
        inputs = {label: nonphysical.reincarnate(physical) for (label, nonphysical), physical in zip(ctx.in_labels, ctx.saved_tensors)}
        f = {nt: ctx.out_values[nt].nonphysical().default_to_nan().reincarnate(g)
             for nt, g in zip(ctx.out_labels, grad_out)
             if g is not None}

        jf_inputs = MultiTensor((FGGMultiShape(ctx.fgg, ctx.out_labels),
                                 FGGMultiShape(ctx.fgg, (el for el, _ in ctx.in_labels))),
                                real_semiring)
        if isinstance(semiring, RealSemiring):
            if ctx.opts.get('j_precompute', False):
                jf = J_precompute_products(ctx.fgg, ctx.out_values, inputs, semiring, jf_inputs)
            else:
                jf = J(ctx.fgg, ctx.out_values, inputs, semiring, jf_inputs)
        elif isinstance(semiring, LogSemiring):
            jf = J_log(ctx.fgg, ctx.out_values, inputs, semiring, jf_inputs)
        else:
            raise ValueError(f'invalid semiring: {semiring}')

        grad_nt = multi_solve(jf, f, transpose=True)
                    
        # Compute gradients of inputs
        grad_t = multi_mv(jf_inputs, grad_nt, transpose=True)
        grad_in = tuple(grad_t[el].project(np.paxes, np.vaxes) for el, np in ctx.in_labels)

        return (None, None, None, None) + grad_in

    @staticmethod
    def apply_to_patterned_tensors(fgg: FGG, opts: Dict, in_labels: Iterable[EdgeLabel], out_labels: Sequence[EdgeLabel], *in_values: PatternedTensor) -> Tuple[PatternedTensor, ...]:
        (nonphysicals, *physicals) = SumProduct.apply(
            fgg, opts,
            tuple((in_label, tensor.nonphysical())
                  for in_label, tensor in zip(in_labels, in_values)),
            out_labels,
            *(tensor.physical for tensor in in_values))
        return tuple(PatternedTensor(opts['semiring'].zeros(fgg.shape(nt)))
                     if nonphysical is None is physical
                     else nonphysical.reincarnate(physical)
                     for nt, nonphysical, physical in zip(out_labels, nonphysicals, physicals))

def sum_products(fgg: FGG, **opts) -> Dict[EdgeLabel, Tensor]:
    opts.setdefault('method',   'fixed-point')
    opts.setdefault('semiring', RealSemiring())
    opts.setdefault('tol',      1e-5) # with float32, 1e-6 can fail
    opts.setdefault('kmax',     1000) # for fixed-point, 100 is too low

    if isinstance(opts['semiring'], BoolSemiring):
        opts['tol'] = 0

    all: Dict[EdgeLabel, PatternedTensor] = {t:cast(FiniteFactor, fgg.factors[t.name]).weights for t in fgg.terminals()}
    for comp in scc(nonterminal_graph(fgg)):

        inputs: Dict[EdgeLabel, PatternedTensor] = {}
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
        comp_values = SumProduct.apply_to_patterned_tensors(fgg, comp_opts, inputs.keys(), comp_labels, *inputs.values())
        all.update(zip(comp_labels, comp_values))
    return {label: t.to_dense() for label, t in all.items()}
        
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
