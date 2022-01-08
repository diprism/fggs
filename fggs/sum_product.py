__all__ = ['sum_product']

from fggs.fggs import FGG, HRG, HRGRule, Interpretation, EdgeLabel, Edge, Node
from fggs.factors import CategoricalFactor
from typing import Callable, Dict, Sequence, Iterable, Tuple, List
from functools import reduce
import collections.abc
import warnings, torch
import torch_semiring_einsum

def _formatwarning(message, category, filename=None, lineno=None, file=None, line=None):
    return '%s:%s: %s: %s' % (filename, lineno, category.__name__, message)
warnings.formatwarning = _formatwarning
Tensor = torch.Tensor; Function = Callable[[Tensor], Tensor]

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
    while any(torch.abs(x1 - x0) > tol) and k <= kmax:
        x0[...], x1[...] = x1, F(x1)
        k += 1
    if k > kmax:
        warnings.warn('maximum iteration exceeded; convergence not guaranteed')

def newton(F: Function, J: Function, x0: Tensor, *, tol: float, kmax: int) -> None:
    k, x1 = 0, x0.clone()
    F0 = F(x0)
    while any(torch.abs(F0) > tol) and k <= kmax:
        JF = J(x0)
        dX = torch.linalg.solve(JF, -F0) if JF.size()[0] > 1 else -F0/JF
        x1[...] = x0 + dX
        x0[...], F0 = x1, F(x1)
        k += 1
    if k > kmax:
        warnings.warn('maximum iteration exceeded; convergence not guaranteed')

def broyden(F: Function, invJ: Tensor, x0: Tensor, *, tol: float, kmax: int) -> None:
    k, x1 = 0, x0.clone()
    F0 = F(x0)
    while any(torch.abs(F0) > tol) and k <= kmax:
        dX = torch.matmul(-invJ, F0)
        x1[...] = x0 + dX
        F1 = F(x1)
        dX, dF = x1 - x0, F1 - F0
        u = (dX - torch.matmul(invJ, dF))/torch.dot(dF, dF)
        invJ += torch.outer(u, dF)
        x0[...], F0 = x1, F1
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


class MultiTensorDict(collections.abc.Mapping):
    """Proxy object returned by MultiTensor.dict."""
    def __init__(self, mt):
        self.mt = mt
    def __iter__(self):
        return iter(self.mt.nt_dict)
    def __len__(self):
        return len(self.mt.nt_dict)
    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        slices, shapes = [], []
        for key in keys:
            (n, k), shape = self.mt.nt_dict[key]
            slices.append(slice(n, k))
            shapes.extend(shape)
        return self.mt._t[slices].reshape(shapes)
    def __setitem__(self, key, val):
        self[key][...] = val
    
    
class MultiTensor:
    """Tensor-like object that concatenates multiple tensors into one."""
    
    # https://pytorch.org/docs/stable/notes/extending.html

    def __init__(self, data: Iterable, nt_dict: Dict = None, **kwargs):
        self._t = torch.as_tensor(data, **kwargs)
        self.nt_dict = nt_dict

    @staticmethod
    def initialize(fgg: FGG, fill_value: float = 0., ndim: int = 1):
        hrg, interp = fgg.grammar, fgg.interp
        n, nt_dict = 0, dict()
        for nonterminal in hrg.nonterminals():
            shape = tuple(interp.domains[label].size() for label in nonterminal.node_labels)
            k = n + (reduce(lambda a, b: a * b, shape) if len(shape) > 0 else 1)
            nt_dict[nonterminal] = ((n, k), shape) # TODO namedtuple(range=(n, k), shape=shape)
            n = k
        return MultiTensor(torch.full(ndim * [n], fill_value=fill_value), nt_dict)

    def clone(self):
        return MultiTensor(self._t.clone(), self.nt_dict)

    def size(self):
        return self._t.size()

    @property
    def dict(self):
        return MultiTensorDict(self)

    def __getitem__(self, key):
        return self._t[key]
    def __setitem__(self, key, value):
        if isinstance(value, MultiTensor):
            value = value._t
        self._t[key] = value

    def __add__(self, other):
        return torch.add(self, other)
    def __sub__(self, other):
        return torch.sub(self, other)
    def __mul__(self, other):
        return torch.mul(self, other)
    def __neg__(self):
        return torch.neg(self)
    def __truediv__(self, other):
        return torch.div(self, other)
    def __gt__(self, other):
        return torch.ge(self, other)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        args = [a._t if isinstance(a, MultiTensor) else a for a in args]
        return func(*args, **kwargs)
    

def F(fgg: FGG, x: MultiTensor, inputs: Dict[EdgeLabel, Tensor]) -> MultiTensor:
    hrg, interp = fgg.grammar, fgg.interp
    Fx = MultiTensor.initialize(fgg, fill_value=-torch.inf)
    for n in hrg.nonterminals():
        for rule in hrg.rules(n):
            tau_rule = sum_product_edges(interp, rule.rhs.nodes(), rule.rhs.edges(), rule.rhs.ext, x.dict, inputs)
            Fx.dict[n] = torch.logaddexp(Fx.dict[n], tau_rule)
    return Fx


def J(fgg: FGG, x: MultiTensor, inputs: Dict[EdgeLabel, Tensor]) -> MultiTensor:
    hrg, interp = fgg.grammar, fgg.interp
    Jx = MultiTensor.initialize(fgg, ndim=2, fill_value=-torch.inf)
    for n in hrg.nonterminals():
        for rule in hrg.rules(n):
            for edge in rule.rhs.edges():
                if edge.label.is_terminal: continue
                ext = rule.rhs.ext + edge.nodes
                edges = set(rule.rhs.edges()) - {edge}
                tau_edge = sum_product_edges(interp, rule.rhs.nodes(), edges, ext, x.dict, inputs)
                Jx.dict[n, edge.label] = torch.logaddexp(Jx.dict[n, edge.label], tau_edge)
    return Jx

def sum_product_edges(interp: Interpretation, nodes: Iterable[Node], edges: Iterable[Edge], ext: Tuple[Node], *inputses: Iterable[Dict[EdgeLabel, Tensor]]) -> Tensor:
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

    connected = set()
    indexing, tensors = [], []
    
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
            tensors.append(torch.log(torch.eye(interp.domains[n.label].size())))
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

        c = torch_semiring_einsum.compile_equation(equation)
        try:
            out = torch_semiring_einsum.log_einsum(c, *tensors, block_size=16)
        except:
            print(equation)
    else:
        out = torch.tensor(0.)

    # Restore any external nodes that were removed.
    if out.ndim < len(ext):
        vshape = [interp.domains[n.label].size() if n in connected else 1 for n in ext]
        eshape = [interp.domains[n.label].size() for n in ext]
        out = out.view(*vshape).expand(*eshape)

    # Multiply in any disconnected internal nodes.
    mul = 1.
    for n in nodes:
        if n not in connected and n not in ext:
            mul *= interp.domains[n.label].size()
    if mul > 1:
        out = out + torch.log(torch.tensor(mul))
        
    return out


def linear(fgg: FGG, inputs: Dict[EdgeLabel, Tensor] = {}) -> MultiTensor:
    """Compute the sum-product of the nonterminals of `fgg`, which is
    linearly recursive given that each nonterminal `x` in `inputs` is
    treated as a terminal with weight `inputs[x]`.
    """
    # to do: update for log semiring
    hrg, interp = fgg.grammar, fgg.interp

    # Check linearity and compute F(0)
    Fx = MultiTensor.initialize(fgg)
    for n in hrg.nonterminals():
        if n in inputs:
            continue
        for rule in hrg.rules(n):
            edges = [e for e in rule.rhs.edges() if e.label.is_nonterminal and e.label not in inputs]
            if len(edges) == 0:
                Fx.dict[n] += sum_product_edges(interp, rule.rhs.nodes(), rule.rhs.edges(), rule.rhs.ext, inputs)
            elif len(edges) > 1:
                raise ValueError('FGG is not linearly recursive')

    x = MultiTensor.initialize(fgg)
    Jx = J(fgg, x, inputs)

    x._t[...] = torch.linalg.solve(torch.eye(Jx.size()[0])-Jx._t, Fx._t)
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
    def forward(ctx, fgg: FGG, opts: Dict, in_labels: Sequence[EdgeLabel], out_labels: Sequence[EdgeLabel], *in_values: Tensor) -> Tuple[Tensor]:
        ctx.fgg = fgg
        opts.setdefault('method', 'fixed-point')
        opts.setdefault('tol',    1e-6)
        opts.setdefault('kmax',   1000)
        ctx.opts = opts
        ctx.in_labels = in_labels
        ctx.out_labels = out_labels
        ctx.save_for_backward(*in_values)

        inputs = dict(zip(in_labels, in_values))

        if opts['method'] == 'linear':
            # To do: make linear() not use custom backward function, and raise an exception here
            out = linear(fgg, inputs)
        else:
            x0 = MultiTensor.initialize(fgg, fill_value=-torch.inf)

            if opts['method'] == 'fixed-point':
                fixed_point(lambda x: F(fgg, x, inputs), x0, tol=opts['tol'], kmax=opts['kmax'])
            elif opts['method'] == 'newton':
                newton(lambda x: F(fgg, x, inputs) - x, lambda x: J(fgg, x, inputs) - torch.eye(x.size()[0]), x0, tol=opts['tol'], kmax=opts['kmax'])
            elif opts['method'] == 'broyden':
                n = x0.size()[0]
                invJ = -torch.eye(n, n)
                broyden(lambda x: F(fgg, x, inputs) - x, invJ, x0, tol=opts['tol'], kmax=opts['kmax'])
            else:
                raise ValueError('unsupported method for computing sum-product')
            out = x0

        ctx.out_values = out
        return tuple(out.dict[nt] for nt in out_labels)

    @staticmethod
    def backward(ctx, *grad_out):
        inputs = dict(zip(ctx.in_labels, ctx.saved_tensors))

        hrg, interp = ctx.fgg.grammar, ctx.fgg.interp

        jf = J(ctx.fgg, ctx.out_values, inputs)
        jf = torch.exp(jf._t + ctx.out_values._t - ctx.out_values._t.unsqueeze(1))
        jf = torch.nan_to_num(jf, 0.) # wherever original jf was -inf

        # Compute F(0) of adjoint grammar
        f = MultiTensor.initialize(ctx.fgg)
        for x, grad_x in zip(ctx.out_labels, grad_out):
            f.dict[x] += grad_x

        # Solve linear system of equations
        grad_nt = MultiTensor.initialize(ctx.fgg)
        grad_nt._t[...] = torch.linalg.solve(torch.eye(jf.size()[0])-jf.T, f._t)

        # Compute gradients of factors
        grad_t = {}
        for el in ctx.in_labels:
            if el.is_terminal:
                grad_t[el] = 0.

        for rule in hrg.all_rules():
            grad_y = grad_nt.dict[rule.lhs]
            for edge in rule.rhs.edges():
                if edge.label in grad_t:
                    ext = rule.rhs.ext + edge.nodes
                    edges = set(rule.rhs.edges()) - {edge}
                    j = sum_product_edges(interp, rule.rhs.nodes(), edges, ext, ctx.out_values.dict, inputs)
                    z_lhs = ctx.out_values.dict[rule.lhs]
                    j = torch.exp(j - z_lhs.reshape(z_lhs.size()+(1,)*edge.label.arity
) + inputs[edge.label])
                    j = torch.nan_to_num(j, 0.) # wherever original j was -inf
                    
                    #grad_t[edge.label] += torch.tensordot(grad_y, j, len(rule.rhs.ext)
                    grad_t[edge.label] += grad_y.reshape(-1) @ j.reshape(-1, *z_lhs.size())

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
        if not isinstance(w, Tensor):
            w = torch.tensor(w, dtype=torch.get_default_dtype())
        in_values.append(torch.log(w))
    out_labels = list(hrg.nonterminals())
    out = SumProduct.apply(fgg, opts, in_labels, out_labels, *in_values)
    return torch.exp(out[out_labels.index(fgg.grammar.start_symbol)])
