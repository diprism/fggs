__all__ = ['sum_product']

from fggs.fggs import FGG, HRG,Interpretation, EdgeLabel, Edge, Node
from fggs.factors import CategoricalFactor
from typing import Callable, Dict, Iterable, Tuple, List
from functools import reduce
import warnings, torch

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
        dX = torch.linalg.solve(JF, -F0) if len(JF) > 1 else -F0/JF
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

class MultiTensor:
    """Tensor-like object that concatenates multiple tensors into one."""
    
    # https://pytorch.org/docs/stable/notes/extending.html

    def __init__(self, data: Iterable, nt_dict: Dict = None, **kwargs):
        self._t = torch.as_tensor(data, **kwargs)
        self.nt_dict = nt_dict

    @staticmethod
    def initialize(fgg: FGG, value: float = 0., dim: int = 1):
        hrg, interp = fgg.grammar, fgg.interp
        n, nt_dict = 0, dict()
        for nonterminal in hrg.nonterminals():
            shape = tuple(interp.domains[label].size() for label in nonterminal.node_labels)
            k = n + (reduce(lambda a, b: a * b, shape) if len(shape) > 0 else 1)
            nt_dict[nonterminal] = ((n, k), shape) # TODO namedtuple(range=(n, k), shape=shape)
            n = k
        return MultiTensor(torch.full(dim * [n], fill_value=value), nt_dict)

    def clone(self):
        return MultiTensor(self._t.clone(), self.nt_dict)

    def size(self):
        return self._t.size()

    def __getitem__(self, key):
        if isinstance(key, (str, EdgeLabel)):
            (n, k), shape = self.nt_dict[key]
            return self._t[n:k].reshape(shape)
        else:
            return self._t[key]

    def __setitem__(self, key, value):
        if isinstance(value, MultiTensor):
            value = value._t
        if isinstance(key, (str, EdgeLabel)):
            (n, k), _ = self.nt_dict[key]
            self._t[n:k] = value.flatten()
        else:
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
        args = [a._t if hasattr(a, '_t') else a for a in args]
        metadata = tuple(a.nt_dict if hasattr(a, 'nt_dict') else a for a in args)
        assert len(metadata) > 0
        return MultiTensor(func(*args, **kwargs), nt_dict=metadata[0])

def F(fgg: FGG, x0: MultiTensor) -> Tensor:
    hrg, interp = fgg.grammar, fgg.interp
    x1 = x0.clone()
    for nonterminal in hrg.nonterminals():
        tau_R = []
        for rule in hrg.rules(nonterminal):
            tau_R.append(sum_product_edges(interp, rule.rhs.ext, rule.rhs.edges(), x0))
        x1[nonterminal] = sum(tau_R)
    return x1

def J(fgg: FGG, x0: MultiTensor) -> Tensor:
    hrg, interp = fgg.grammar, fgg.interp
    JF = torch.full(2 * list(x0._t.shape), fill_value=0.)
    ############# TODO
    p, q = [0, 0], [0, 0]
    for nt_num in hrg.nonterminals():
        p[0] = p[1]
        a, b = x0.nt_dict[nt_num][0]
        p[1] += b - a
        q_ = q[:] # q[1] = 0
        for nt_den in hrg.nonterminals():
            q[0] = q[1]
            a, b = x0.nt_dict[nt_den][0]
            q[1] += b - a
    #############
            tau_R = [torch.tensor(0.)]
            for rule in hrg.rules(nt_num):
                if len(rule.rhs.nodes()) > 26:
                    raise Exception('cannot assign an index to each node')
                Xi_R = {id: chr(ord('a') + i) for i, id in enumerate(rule.rhs._node_ids)}
                nt_loc, tensors, indexing = [], [], []
                for i, edge in enumerate(rule.rhs.edges()):
                    indexing.append([Xi_R[node.id] for node in edge.nodes])
                    if edge.label.is_nonterminal:
                        tensors.append(x0[edge.label])
                        nt_loc.append((i, edge.label.name))
                    elif isinstance(interp.factors[edge.label], CategoricalFactor):
                        weights = interp.factors[edge.label]._weights
                        if not isinstance(weights, Tensor):
                            weights = torch.tensor(weights)
                        tensors.append(weights)
                    else:
                        raise TypeError(f'cannot compute sum-product of FGG with factor {interp.factors[edge.label]}')
                external = [Xi_R[node.id] for node in rule.rhs.ext]
                # TODO sum_product_edges for each term in the product rule
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

def sum_product_edges(interp: Interpretation, ext: Tuple[Node], edges: Iterable[Edge], sumprod: MultiTensor = None) -> Tensor:
    # The sum-product of an empty set of edges is 1
    if len(edges) == 0:
        return 1.
    
    # Each node corresponds to an index, so choose a letter for each
    nodes = set()
    for edge in edges:
        nodes.update(edge.nodes)
    if len(nodes) > 26:
        raise Exception('cannot assign an index to each node')
    node_to_index = {node: chr(ord('a') + i) for i, node in enumerate(nodes)}
    
    indexing, tensors = [], []
    for edge in edges:
        indexing.append([node_to_index[node] for node in edge.nodes])
        if edge.label.is_nonterminal:
            tensors.append(sumprod[edge.label])
        elif isinstance(interp.factors[edge.label], CategoricalFactor):
            weights = interp.factors[edge.label]._weights
            if not isinstance(weights, Tensor):
                weights = torch.tensor(weights)
            tensors.append(weights)
        else:
            raise TypeError(f'cannot compute sum-product of FGG with factor {interp.factors[edge.label]}')
    equation = ','.join([''.join(indices) for indices in indexing]) + '->'
    
    # If an external node has no edges, einsum will complain, so remove it.
    external = [node_to_index[node] for node in ext if node in nodes]
    equation += ''.join(external)
    
    out = torch.einsum(equation, *tensors)
    
    # Restore any external nodes that were removed.
    if 0 < len(external) < len(nodes):
        vshape = [interp.domains[n.label].size() if n in nodes else 1 for n in ext]
        eshape = [interp.domains[n.label].size() for n in ext]
        out = out.view(*vshape).expand(*eshape)
        
    return out

def linear(fgg: FGG, sumprod: MultiTensor) -> Tensor:
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

    n = sumprod.size()[0]
    f = torch.zeros(n)
    jf = torch.zeros(n, n)
    for x in hrg.nonterminals():
        (xi, xj), _ = sumprod.nt_dict[x]
        # Compute JF(0)
        for y in hrg.nonterminals():
            (yi, yj), _ = sumprod.nt_dict[y]
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

    sumprod._t[...] = torch.linalg.solve(torch.eye(n)-jf, f)

def sum_product(fgg: FGG, *, method: str = 'fixed-point', tol: float = 1e-6, kmax: int = 1000) -> Tensor:
    """Compute the sum-product of an FGG.

    - fgg: The FGG to compute the sum-product of.
    - method: What method to use ('linear', 'fixed-point', 'newton', 'broyden').
    - tol: Iterative algorithms terminate when the L∞ distance between consecutive iterates is below tol.
    - kmax: Number of iterations after which iterative algorithms give up.
    """
    x0 = MultiTensor.initialize(fgg)
    if method == 'linear':
        linear(fgg, x0)
    elif method == 'fixed-point':
        fixed_point(lambda x: F(fgg, x), x0, tol=tol, kmax=kmax)
    elif method == 'newton':
        newton(lambda x: F(fgg, x) - x, lambda x: J(fgg, x), x0, tol=tol, kmax=kmax)
    elif method == 'broyden':
        # Broyden's method may fail to converge without a sufficient
        # initial approximation of the Jacobian. If the method doesn't
        # converge within N iteration(s), perturb the initial approximation.
        # Source: Numerical Recipes in C: the Art of Scientific Computing
        n = x0.size()[0]
        invJ = -torch.eye(n, n)
        broyden(lambda x: F(fgg, x) - x, invJ, x0, tol=tol, kmax=kmax)
    else: raise ValueError('unsupported method for computing sum-product')
    return x0[fgg.grammar.start_symbol]
