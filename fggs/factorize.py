__all__ = ['factorize_rule', 'factorize_hrg', 'factorize_fgg']

from fggs import fggs, utils
import copy

def add_node(graph, v):
    if v not in graph:
        graph[v] = set()

def remove_node(graph, v):
    for u in graph[v]:
        graph[u].discard(v)
    del graph[v]

def add_edge(graph, u, v):
    graph[u].add(v)
    graph[v].add(u)

def contract_edge(graph, u, v):
    for vn in graph[v]:
        if vn != u:
            add_edge(graph, u, vn)
    remove_node(graph, v)

def copy_graph(graph):
    return {u: set(graph[u]) for u in graph}

def make_clique(graph, nodes):
    for v1 in nodes:
        for v2 in nodes:
            if v1 != v2:
                add_edge(graph, v1, v2)

def count_fillin(graph, u):
    """How many edges would be needed to make v a clique."""
    count = 0
    for v1 in graph[u]:
        for v2 in graph[u]:
            if v1 != v2 and v2 not in graph[v1]:
                count += 1
    return count//2

def is_clique(graph, vs):
    for v1 in vs:
        for v2 in vs:
            if v1 != v2 and v2 not in graph[v1]:
                return False
    return True

def simplicial(graph, v):
    return is_clique(graph, graph[v])

def almost_simplicial(graph, v):
    for u in graph[v]:
        if is_clique(graph, graph[v] - {u}):
            return True
    return False

def eliminate_node(graph, v):
    if v not in graph:
        raise KeyError("node not in graph")
    make_clique(graph, graph[v])
    remove_node(graph, v)

def min_fill(graph):
    """Min-fill upper-bound."""
    graph = copy_graph(graph)
    dmax = 0
    order = []
    while len(graph) > 0:
        u = min(graph, key=lambda u: count_fillin(graph, u))
        dmax = max(dmax, len(graph[u]))
        eliminate_node(graph, u)
        order.append(u)
    return dmax, order

def minor_min_width(graph):
    """Minor-min-width lower-bound."""
    graph = copy_graph(graph)
    dmax = 0
    while len(graph) > 0:
        # Let v be a node of minimum degree
        v = min(graph, key=lambda v: len(graph[v]))
        dmax = max(dmax, len(graph[v]))

        nb = graph[v]
        assert v not in nb
        if len(nb) > 0:
            # Let u \in N(v) such that the degree of u is minimum in N(v)
            #u = min(nb, key=lambda u: len(graph[u])) # min-d (this is what Gogate's code does)
            u = min(nb, key=lambda u: len(graph[u] & nb)) # least-c (Bodlaender and Koster say this works better)
            contract_edge(graph, u, v)
        else:
            # v has degree zero, so the rest will too
            break
    return dmax

upper_bound = min_fill
lower_bound = minor_min_width

def quickbb(graph):
    """Gogate and Dechter, A complete anytime algorithm for treewidth. UAI
    2004. http://arxiv.org/pdf/1207.4109.pdf
    Code: https://github.com/dechterlab/quickbb

    Given a permutation of the nodes (called an elimination ordering),
    for each node, remove the node and make its neighbors into a clique.
    The maximum degree of the nodes at the time of their elimination is
    the width of the tree decomposition corresponding to that ordering.
    The treewidth of the graph is the minimum over all possible
    permutations.

    This function implements some but not all of the heuristics in the
    original paper. Namely, it implements Section 5.1 but not 5.2-3 or 6.

    Arguments:
    graph: dict[node, Iterable[node]]
    """

    # Make into an undirected graph without self-loops
    g1 = {}
    for u in graph:
        g1[u] = set()
    for u in graph:
        for v in graph[u]:
            if u != v:
                g1[u].add(v)
                g1[v].add(u)
    graph = g1

    best_ub = best_order = None
    
    def bb(graph, order, sep, f, g):
        """Branch-and-bound search.

        graph: graph with some nodes eliminated
        order: the eliminated nodes (in order)
        sep: neighbors of order[-1]
        f: lower bound on treewidth so far
        g: treewidth so far
        """
        nonlocal best_ub, best_order
        if len(graph) < 2:
            if f < best_ub:
                assert f == g
                best_ub = f
                best_order = list(order) + list(graph)
        else:
            # Build a list of nodes to try eliminating next
            vs = []
            for v in graph:
                # Graph reduction (Section 5.1): very important
                if (simplicial(graph, v) or
                    almost_simplicial(graph, v) and len(graph[v]) <= lb):
                    vs = [v]
                    break
                elif v not in sep: # Section 5.2
                    vs.append(v)

            # to do: sort by min-fill?
            #vs.sort(key=lambda u: count_fillin(graph, graph[u]))
            
            for v in vs:
                graph1 = copy_graph(graph)
                sep1 = graph[v]
                eliminate_node(graph1, v)
                order1 = order + [v]
                # treewidth for current order so far
                g1 = max(g, len(graph[v])) 
                # lower bound given where we are
                f1 = max(g, lower_bound(graph1)) 
                if f1 < best_ub:
                    bb(graph1, order1, sep1, f1, g1)

    order = []
    best_ub, best_order = upper_bound(graph)
    lb = lower_bound(graph)
    if lb < best_ub:
        bb(graph, order, [], lb, 0)

    return best_ub, best_order

def connected_components(g, s=frozenset()):
    """Find the connected components of g \ s."""

    comps = []
    nodes = set(g) - s
    while len(nodes) > 0:
        comp = set()
        agenda = {nodes.pop()}
        while len(agenda) > 0:
            v = agenda.pop()
            comp.add(v)
            agenda.update(g[v] - comp - s)
        nodes -= comp
        comps.append(comp)
    return comps

def acb(g):
    comptrees = []
    for c in connected_components(g):
        c = {v:g[v] for v in c}
        ub, _ = upper_bound(c)
        if ub == 0:
            unrooted = {}
            add_node(unrooted, frozenset(c.keys()))
            return unrooted
        for k in range(1, ub+1):
            comptree = acb_connected(c, k)
            if comptree is not False:
                comptrees.append(comptree)
                break
        else:
            assert False, f"No tree decomposition with width bounded by upper bound of {ub}"
    if len(comptrees) == 1:
        tree = comptrees[0]
    else:
        tree = (frozenset(), comptrees)

    # un-root the tree decomposition
    unrooted = {}
    def build(node):
        bag, children = node
        add_node(unrooted, bag)
        for child in children:
            add_edge(unrooted, bag, build(child))
        return bag
    build(tree)

    return unrooted

def acb_connected(g, k):
    """
    Complexity of finding embeddings in a $k$-tree.
    Stefan Arnborg, Derek G. Corneil, and Andrzej Proskurowski.
    https://epubs.siam.org/doi/abs/10.1137/0608024
    
    """
    import itertools
    
    assert len(connected_components(g)) == 1
    
    if len(g) <= k+1:
        return (frozenset(g), [])

    chart = {}
    # for each set C_i of k vertices in G
    for i in itertools.combinations(g, k):
        i = frozenset(i)
        # if C_i is a separator of G
        comps = connected_components(g, i)
        if len(comps) > 1:
            chart[i] = {}
            # C_i^j is (connected component of G \ C_i) | (complete graph on C_i)
            for c in comps:
                j = frozenset(c|i)
                chart[i][j] = None
    if len(chart) == 0:
        return False

    dead = set()
    
    # for each graph C_i^j in increasing order of size
    bysize = sorted((len(j), i, j) for i in chart for j in chart[i])
    for (h,i,j) in bysize:
        # A graph of size k+1 (or less) is a partial k-tree
        if h <= k+1:
            chart[i][j] = (j, [])
        else:
            for v in j - i:
                # examine all k-vertex separators C_m in C_i | {v},
                # except C_i itself (not stated in pseudocode; see Lemma 4.4)
                bag = i | {v}
                children = []
                union = set()
                for u in i:
                    m = bag - {u}
                    # consider all C_m^l in C_i^j which are partial k-trees
                    if m in chart:
                        for l in chart[m]:
                            lm = l-m
                            if lm.issubset(j-bag) and chart[m][l] not in [None, False]:
                                # the C_m^l \ C_m are disjoint but there could be repeats
                                if len(lm & union) == 0:
                                    union |= lm
                                    children.append(chart[m][l])
                                else:
                                    assert lm.issubset(union)
                # if the C_m^l \ C_m partition C_i^j - C_i - {v}, then set answer to YES
                if union == j-bag:
                    chart[i][j] = (bag, children)
                    break

        # if no answer was set for C_i^j, set answer to NO
        if chart[i][j] is None:
            chart[i][j] = False
            dead.add(i)
        # if each separator C_i has a C_i^j with answer NO, return NO
        if len(dead) == len(chart):
            return False
        # if G has a separator C_i such that all C_i^j graph have YES, return YES
        if all(chart[i][j] not in [None, False] for j in chart[i]):
            return (i, [chart[i][j] for j in chart[i]])
    assert False, f"No decomposition found with width {k}"

def tree_decomposition_from_order(graph, order):
    tree = {}
    def build(order):
        v = order[0]
        clique = set(graph[v])
        eliminate_node(graph, v)
        if len(clique) < len(graph):
            build(order[1:])
            for tv in tree:
                if clique.issubset(tv):
                    break
            else:
                assert False
        else:
            tv = None
        tnew = frozenset(clique|{v})
        add_node(tree, tnew)
        if tv is not None:
            add_edge(tree, tv, tnew)
    if len(order) == 0:
        add_node(tree, frozenset())
    else:
        build(order)
    return tree
    
def tree_decomposition(graph, method='min_fill'):
    if method == 'quickbb':
        _, order = quickbb(graph)
        return tree_decomposition_from_order(graph, order)
    elif method == 'min_fill':
        _, order = min_fill(graph)
        return tree_decomposition_from_order(graph, order)
    elif method == 'acb':
        return acb(graph)
    else:
        raise ValueError("unknown method '{method}'")

def factorize_rule(rule, method='min_fill', labels=None):
    """Factorize a rule into one or more smaller rules, hopefully with
    lower maximum treewidth.

    Arguments:
    - rule: the HRGRule to factorize
    - method: tree decomposition method
    - labels: The set of EdgeLabel names to avoid. New EdgeLabels are added to the set.
    """
    rhs = rule.rhs
    if labels is None:
        labels = set()
    labels.add(rule.lhs)
    labels.update(rule.rhs.nonterminals())
    
    # Find tree decomposition of rhs
    g = {}
    for v in rhs.nodes():
        g[v] = set()
    for nodes in [e.nodes for e in rhs.edges()] + [rhs.ext]:
        for u in nodes:
            for v in nodes:
                if u != v:
                    g[u].add(v)
                    g[v].add(u)
    t = tree_decomposition(g, method=method)

    # The root bag is the one that contains all the externals
    ext = set(rhs.ext)
    for bag in t:
        if ext.issubset(bag):
            root = bag
            break
    else:
        assert False, "There must be a bag containing all external nodes"

    newrules = []
    
    def visit(bag, parent):

        # lhs and external nodes
        if parent is None:
            ext = rule.rhs.ext
            lhs = rule.lhs
        else:
            ext = list(bag & parent)
            lhs = fggs.EdgeLabel(utils.unique_label_name(rule.lhs.name, labels),
                                 is_nonterminal=True,
                                 node_labels=tuple([v.label for v in ext]))
            labels.add(lhs)
        
        # rhs
        rhs = fggs.Graph()
        for v in bag:
            rhs.add_node(v)
        rhs.ext = ext

        # terminal edges and existing nonterminal edges
        for e in rule.rhs.edges():
            if (bag.issuperset(e.nodes) and
                (parent is None or not parent.issuperset(e.nodes))):
                rhs.add_edge(e)

        # new nonterminal edges and recurse on children
        for n in t[bag]:
            if n != parent:
                child_lhs, child_ext = visit(n, bag)
                rhs.add_edge(fggs.Edge(child_lhs, child_ext))

        newrule = fggs.HRGRule(lhs, rhs)
        newrules.append(newrule)
        return (lhs, ext)
    
    visit(root, None)
    
    return newrules

def factorize_hrg(g, method='min_fill'):
    """Factorize a HRG's rules into smaller rules, hopefully with
    lower maximum treewidth.
    """
    gnew = fggs.HRG(g.start)
    labels = set(g.edge_labels())
    for r in g.all_rules():
        for rnew in factorize_rule(r, method=method, labels=labels):
            gnew.add_rule(rnew)
    return gnew

def factorize_fgg(g, method='min_fill'):
    """Factorize a FGG's rules into smaller rules, hopefully with
    lower maximum treewidth.
    """
    gnew = fggs.FGG.from_hrg(factorize_hrg(g))
    gnew.factors = g.factors
    gnew.domains = g.domains
    return gnew
    
