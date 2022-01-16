__all__ = ['sum_product']

from fggs.fggs import FGG, HRG, HRGRule, Interpretation, EdgeLabel, Edge, Node
from fggs.factors import CategoricalFactor
from fggs.semirings import *
from typing import Callable, Dict, Mapping, Sequence, Iterable, Tuple, List, Set, Union, cast, Optional
from functools import reduce
import torch_semiring_einsum # type: ignore
import warnings
import torch
from torch import Tensor
Function = Callable[[Tensor], Tensor]


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

    
def fixed_point(F: Function, x0: Tensor, *, tol: float, kmax: int) -> None:
    k, x1 = 0, F(x0)
    while (tol == 0 and x1 != x0 or tol > 0 and any(torch.abs(x1 - x0) > tol)) and k <= kmax:
        x0.copy_(x1)
        x1.copy_(F(x1))
        k += 1
    if k > kmax:
        warnings.warn('maximum iteration exceeded; convergence not guaranteed')

def newton(F: Function, J: Function, x0: Tensor, *, tol: float, kmax: int) -> None:
    k, x1 = 0, x0.clone()
    F0 = F(x0)
    n = x0.size()[0]
    while any(torch.abs(F0) > tol) and k <= kmax:
        JF = J(x0)
        dX = torch.linalg.solve(JF, -F0) if n > 1 else (-F0/JF).reshape(n) # type: ignore
        x1.copy_(x0 + dX)
        x0.copy_(x1)
        F0 = F(x1)
        k += 1
    if k > kmax:
        warnings.warn('maximum iteration exceeded; convergence not guaranteed')

def broyden(F: Function, invJ: Tensor, x0: Tensor, *, tol: float, kmax: int) -> None:
    k, x1 = 0, x0.clone()
    F0 = F(x0)
    while any(torch.abs(F0) > tol) and k <= kmax:
        dX = torch.matmul(-invJ, F0) # type: ignore
        x1.copy_(x0 + dX)
        F1 = F(x1)
        dX, dF = x1 - x0, F1 - F0
        u = (dX - torch.matmul(invJ, dF))/torch.dot(dF, dF)
        invJ += torch.outer(u, dF)
        x0.copy_(x1)
        F0 = F1
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


class MultiTensorDict(Mapping[EdgeLabel, Tensor]):
    """Proxy object returned by MultiTensor.dict."""
    def __init__(self, mt):
        self.mt = mt
    def __iter__(self):
        return iter(self.mt.nt_dict)
    def __len__(self) -> int:
        return len(self.mt.nt_dict)
    def __getitem__(self, keys: Union[EdgeLabel, Tuple[EdgeLabel]]) -> Tensor:
        if not isinstance(keys, tuple):
            keys = (keys,)
        slices, shapes = [], []
        for key in keys:
            (n, k), shape = self.mt.nt_dict[key]
            slices.append(slice(n, k))
            shapes.extend(shape)
        return self.mt[slices].reshape(shapes)
    def __setitem__(self, key: Union[EdgeLabel, Tuple[EdgeLabel]], val: Tensor):
        self[key].copy_(val)
    
    
class MultiTensor(Tensor):
    """Tensor-like object that concatenates multiple tensors into one."""
    
    # https://pytorch.org/docs/stable/notes/extending.html

    def __new__(cls, data: Iterable, nt_dict: Dict = None, **kwargs):
        return super().__new__(cls, data, **kwargs) # type: ignore
    def __init__(self, data: Iterable, nt_dict: Dict = None, **kwargs):
        self.nt_dict = nt_dict

    @staticmethod
    def initialize(fgg: FGG, fill_value: float = 0., ndim: int = 1):
        hrg, interp = fgg.grammar, fgg.interp
        n, nt_dict = 0, dict()
        for nonterminal in hrg.nonterminals():
            shape = tuple(interp.shape(nonterminal))
            k = n + (reduce(lambda a, b: a * b, shape) if len(shape) > 0 else 1)
            nt_dict[nonterminal] = ((n, k), shape) # TODO namedtuple(range=(n, k), shape=shape)
            n = k
        mt = torch.full(ndim * [n], fill_value=fill_value).as_subclass(MultiTensor)
        mt.nt_dict = nt_dict
        return mt

    @property
    def dict(self) -> MultiTensorDict:
        return MultiTensorDict(self)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        res = super().__torch_function__(func, types, args, kwargs)
            
        # Setting res.nt_dict is difficult in general,
        # but we only need to do it correctly for copy_() and clone().
        if func is Tensor.clone and hasattr(args[0], 'nt_dict'):
            res.nt_dict = args[0].nt_dict
        elif isinstance(res, MultiTensor) and not hasattr(res, 'nt_dict'):
            res.nt_dict = None
        return res

def F(fgg: FGG, x: MultiTensor, inputs: Mapping[EdgeLabel, Tensor], semiring) -> MultiTensor:
    hrg, interp = fgg.grammar, fgg.interp
    Fx = MultiTensor.initialize(fgg, fill_value=semiring.zero)
    for n in hrg.nonterminals():
        for rule in hrg.rules(n):
            tau_rule = sum_product_edges(interp, rule.rhs.nodes(), rule.rhs.edges(), rule.rhs.ext, x.dict, inputs, semiring=semiring)
            Fx.dict[n] = semiring.add(Fx.dict[n], tau_rule)
    return Fx


def J(fgg: FGG, x: MultiTensor, inputs: Mapping[EdgeLabel, Tensor],
      J_terminals: Optional[Dict[Tuple[EdgeLabel,EdgeLabel],Tensor]] = None) -> MultiTensor:
    """The Jacobian of F(semiring=RealSemiring)."""
    hrg, interp = fgg.grammar, fgg.interp
    Jx = MultiTensor.initialize(fgg, ndim=2)
    for n in hrg.nonterminals():
        for rule in hrg.rules(n):
            for edge in rule.rhs.edges():
                if edge.label.is_terminal and J_terminals is None: continue
                ext = rule.rhs.ext + edge.nodes
                edges = set(rule.rhs.edges()) - {edge}
                tau_edge = sum_product_edges(interp, rule.rhs.nodes(), edges, ext, x.dict, inputs, semiring=RealSemiring)
                if edge.label.is_terminal:
                    assert J_terminals is not None
                    if (n, edge.label) in J_terminals:
                        J_terminals[n, edge.label] += tau_edge
                    else:
                        J_terminals[n, edge.label] = tau_edge
                else:
                    Jx.dict[n, edge.label] += tau_edge
    return Jx


def smooth_log_softmax(a, dim):
    # Clip large logits to a large, but not huge, number. This
    # treats all large numbers as equal, which "smooths" the
    # softmax and helps gradient to flow to all rules.
    a = torch.min(a, torch.tensor(1000.))
    return a.log_softmax(dim=dim).nan_to_num()


def J_log(fgg: FGG, x: MultiTensor, inputs: Dict[EdgeLabel, Tensor],
          J_terminals: Optional[Dict[Tuple[EdgeLabel,EdgeLabel],Tensor]] = None) -> MultiTensor:
    """The Jacobian of F(semiring=LogSemiring)."""
    hrg, interp = fgg.grammar, fgg.interp
    Jx = MultiTensor.initialize(fgg, ndim=2)
    for n in hrg.nonterminals():
        rules = list(hrg.rules(n))
        tau_rules: Union[List[Tensor], Tensor]
        tau_rules = []
        for rule in rules:
            tau_rule = sum_product_edges(interp, rule.rhs.nodes(), rule.rhs.edges(), rule.rhs.ext, x.dict, inputs, semiring=LogSemiring)
            tau_rules.append(tau_rule)
        tau_rules = torch.stack(tau_rules, dim=0)
        tau_rules = smooth_log_softmax(tau_rules, dim=0)
        for rule, tau_rule in zip(rules, tau_rules):
            for edge in rule.rhs.edges():
                if edge.label.is_terminal and J_terminals is None: continue
                ext = rule.rhs.ext + edge.nodes
                tau_edge = sum_product_edges(interp, rule.rhs.nodes(), rule.rhs.edges(), ext, x.dict, inputs, semiring=LogSemiring)
                tau_edge_size = tau_edge.size()
                tau_edge = tau_edge.reshape(tau_rule.size() + (-1,))
                tau_edge = smooth_log_softmax(tau_edge, dim=-1)
                tau_edge += tau_rule.unsqueeze(-1)
                tau_edge = tau_edge.reshape(tau_edge_size)
                if edge.label.is_terminal:
                    assert J_terminals is not None
                    if (n, edge.label) in J_terminals:
                        J_terminals[n, edge.label] += torch.exp(tau_edge)
                    else:
                        J_terminals[n, edge.label] = torch.exp(tau_edge)
                else:
                    Jx.dict[n, edge.label] += torch.exp(tau_edge)
    return Jx

def sum_product_edges(interp: Interpretation, nodes: Iterable[Node], edges: Iterable[Edge], ext: Sequence[Node], *inputses: Mapping[EdgeLabel, Tensor], semiring) -> Tensor:
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
            tensors.append(semiring.from_int(torch.eye(interp.domains[n.label].size())))
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
        out = semiring.einsum(compiled, *tensors, block_size=16)
    else:
        out = torch.tensor(semiring.one)

    # Restore any external nodes that were removed.
    if out.ndim < len(ext):
        vshape = [interp.domains[n.label].size() if n in connected else 1 for n in ext]
        eshape = interp.shape(ext)
        out = out.view(*vshape).expand(*eshape)

    # Multiply in any disconnected internal nodes.
    mul = 1.
    for n in nodes:
        if n not in connected and n not in ext:
            mul *= interp.domains[n.label].size()
    if mul > 1:
        out = semiring.mul(out, semiring.from_int(mul))

    return out


def linear(fgg: FGG, inputs: Mapping[EdgeLabel, Tensor] = {}) -> MultiTensor:
    """Compute the sum-product of the nonterminals of `fgg`, which is
    linearly recursive given that each nonterminal `x` in `inputs` is
    treated as a terminal with weight `inputs[x]`.
    """
    hrg, interp = fgg.grammar, fgg.interp

    # Check linearity and compute F(0)
    Fx = MultiTensor.initialize(fgg)
    for n in hrg.nonterminals():
        if n in inputs:
            continue
        for rule in hrg.rules(n):
            edges = [e for e in rule.rhs.edges() if e.label.is_nonterminal and e.label not in inputs]
            if len(edges) == 0:
                Fx.dict[n] += sum_product_edges(interp, rule.rhs.nodes(), rule.rhs.edges(), rule.rhs.ext, inputs, semiring=RealSemiring)
            elif len(edges) > 1:
                raise ValueError('FGG is not linearly recursive')

    x = MultiTensor.initialize(fgg)
    Jx = J(fgg, x, inputs)

    x.copy_(torch.linalg.solve(torch.eye(Jx.size()[0])-Jx, Fx))
    return x


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
        opts.setdefault('semiring', RealSemiring)
        opts.setdefault('tol',      1e-6)
        opts.setdefault('kmax',     1000)
        method, semiring = opts['method'], opts['semiring']
        if semiring is BoolSemiring:
            opts['tol'] = 0
        
        ctx.opts = opts
        ctx.in_labels = in_labels
        ctx.out_labels = out_labels
        ctx.save_for_backward(*in_values)

        inputs = dict(zip(in_labels, in_values))

        if method == 'linear':
            # To do: make linear() not use custom backward function, and raise an exception here
            if semiring is not RealSemiring:
                raise NotImplementedError()
            out = linear(fgg, inputs)
        else:
            x0 = MultiTensor.initialize(fgg, fill_value=semiring.zero)
            n = x0.size()[0]

            if method == 'fixed-point':
                fixed_point(lambda x: F(fgg, cast(MultiTensor, x), inputs, semiring),
                            x0, tol=opts['tol'], kmax=opts['kmax'])
            elif method == 'newton':
                if semiring is not RealSemiring:
                    raise NotImplementedError()
                newton(lambda x: F(fgg, cast(MultiTensor, x), inputs, RealSemiring) - x,
                       lambda x: J(fgg, cast(MultiTensor, x), inputs) - torch.eye(n),
                       x0, tol=opts['tol'], kmax=opts['kmax'])
            elif method == 'broyden':
                if semiring is not RealSemiring:
                    raise NotImplementedError()
                broyden(lambda x: F(fgg, cast(MultiTensor, x), inputs, RealSemiring) - x,
                        -torch.eye(n), # type: ignore
                        x0, tol=opts['tol'], kmax=opts['kmax'])
            else:
                raise ValueError('unsupported method for computing sum-product')
            out = x0

        ctx.out_values = out
        return tuple(out.dict[nt] for nt in out_labels)

    @staticmethod
    def backward(ctx, *grad_out):
        inputs = dict(zip(ctx.in_labels, ctx.saved_tensors))
        hrg, interp = ctx.fgg.grammar, ctx.fgg.interp
        semiring = ctx.opts['semiring']

        jf_terminals: Dict[Tuple[EdgeLabel,EdgeLabel],Tensor] = {}
        if semiring is RealSemiring:
            jf = J(ctx.fgg, ctx.out_values, inputs, jf_terminals)
        elif semiring is LogSemiring:
            jf = J_log(ctx.fgg, ctx.out_values, inputs, jf_terminals)
        else:
            raise ValueError(f'invalid semiring: {semiring}')
        
        f = MultiTensor.initialize(ctx.fgg)
        for x, grad_x in zip(ctx.out_labels, grad_out):
            f.dict[x] += grad_x

        # Solve linear system of equations
        grad_nt = MultiTensor.initialize(ctx.fgg)
        try:
            assert torch.all(f >= 0.)
            grad_nt[...] = torch.linalg.solve(torch.eye(jf.size()[0])-jf.T, f)
            failed = not torch.all(grad_nt >= 0.) # negative or NaN
        except RuntimeError:
            failed = True
        if failed:
            warnings.warn('SumProduct.backward(): torch.linalg.solve failed; using fixed-point iteration')
            grad_nt.fill_(0.)
            fixed_point(lambda g: torch.nan_to_num(f + jf.T @ g),
                        grad_nt,
                        tol=ctx.opts.get('tol', 1e-6),
                        kmax=ctx.opts.get('kmax', 1000))
                    
        # Compute gradients of factors
        grad_t = {t:0. for t in hrg.terminals()}
        for y, t in jf_terminals:
            delta_grad_t = tensordot(grad_nt.dict[y], jf_terminals[y,t], y.arity)
            delta_grad_t = delta_grad_t.nan_to_num() # needed when grad_nt[y] = inf, jf[y,t] = 0
            grad_t[t] += delta_grad_t

        grad_in = tuple(grad_t[el] if el.is_terminal else grad_nt[el] for el in ctx.in_labels)
        
        return (None, None, None, None) + grad_in


def sum_product(fgg: FGG, **opts) -> Tensor:
    """Compute the sum-product of an FGG.
    
    - fgg: The FGG to compute the sum-product of.
    - method: What method to use ('linear', 'fixed-point', 'newton', 'broyden').
    - tol: Iterative algorithms terminate when the L∞ distance between consecutive iterates is below tol.
    - kmax: Number of iterations after which iterative algorithms give up.
    """

    hrg, interp = fgg.grammar, fgg.interp
    in_labels = list(hrg.terminals())
    in_values = []
    for t in in_labels:
        w = interp.factors[t].weights
        w = torch.as_tensor(w, dtype=torch.get_default_dtype())
        in_values.append(w)
    out_labels = list(hrg.nonterminals())
    out = SumProduct.apply(fgg, opts, in_labels, out_labels, *in_values)
    return out[out_labels.index(fgg.grammar.start_symbol)]
