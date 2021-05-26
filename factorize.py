__all__ = ['factorize_rule', 'factorize']

import fgg_representation as fggs

def add_node(graph, v):
    if v not in graph:
        graph[v] = set()

def remove_node(graph, v):
    for u in graph:
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

def count_fillin(graph, nodes):
    """How many edges would be needed to make v a clique."""
    count = 0
    for v1 in nodes:
        for v2 in nodes:
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
        if is_clique(graph, set(graph[v]) - {u}):
            return True
    return False

def eliminate_node(graph, v):
    if v not in graph:
        raise KeyError("node not in graph")
    make_clique(graph, graph[v])
    remove_node(graph, v)

def upper_bound(graph):
    """Min-fill."""
    graph = copy_graph(graph)
    dmax = 0
    order = []
    while len(graph) > 0:
        #d, u = min((len(graph[u]), u) for u in graph) # min-width
        u = min(graph, key=lambda u: count_fillin(graph, graph[u]))
        dmax = max(dmax, len(graph[u]))
        eliminate_node(graph, u)
        order.append(u)
    return dmax, order

def lower_bound(graph):
    """Minor-min-width"""
    graph = copy_graph(graph)
    dmax = 0
    while len(graph) > 0:
        # pick node of minimum degree
        u = min(graph, key=lambda u: len(graph[u]))
        dmax = max(dmax, len(graph[u]))

        # Gogate and Dechter: minor-min-width
        nb = set(graph[u]) - {u}
        if len(nb) > 0:
            v = min(nb, key=lambda u: len(set(graph[u]) & nb))
            contract_edge(graph, u, v)
        else:
            remove_node(graph, u)
    return dmax

class Bag:
    def __init__(self, nodes):
        self.nodes = set(nodes)
    def __str__(self):
        return str(self.nodes)
    def __repr__(self):
        return f'Bag({self.nodes})'

def quickbb(graph):
    """Gogate and Dechter, A complete anytime algorithm for treewidth. UAI
    2004. http://arxiv.org/pdf/1207.4109.pdf

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
    
    def bb(graph, order, f, g):
        """Branch-and-bound search.

        graph: graph with some nodes eliminated
        order: the eliminated nodes (in order)
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
                else:
                    vs.append(v)

            for v in vs:
                graph1 = copy_graph(graph)
                exclude1 = set(graph[v])
                eliminate_node(graph1, v)
                order1 = order + [v]
                # treewidth for current order so far
                g1 = max(g, len(graph[v])) 
                # lower bound given where we are
                f1 = max(g, lower_bound(graph1)) 
                if f1 < best_ub:
                    bb(graph1, order1, f1, g1)

    order = []
    best_ub, best_order = upper_bound(graph)
    lb = lower_bound(graph)
    if lb < best_ub:
        bb(graph, order, lb, 0)

    # Build the tree decomposition
    tree = {}
    def build(order):
        v = order[0]
        clique = set(graph[v])
        eliminate_node(graph, v)
        if len(clique) < len(graph):
            build(order[1:])
            for tv in tree:
                if clique.issubset(tv.nodes):
                    break
            else:
                assert False
        else:
            tv = None
        tnew = Bag(clique|{v})
        add_node(tree, tnew)
        if tv is not None:
            add_edge(tree, tv, tnew)
    build(best_order)
    return tree

def factorize_rule(rule):
    """Factorize a rule into one or more smaller rules, hopefully with
    lower maximum treewidth.
    """
    rhs = rule.rhs()
    
    # Find tree decomposition of rhs
    g = {}
    for v in rhs.nodes():
        g[v] = set()
    for nodes in [e.nodes() for e in rhs.edges()] + [rhs.ext()]:
        for u in nodes:
            for v in nodes:
                if u != v:
                    g[u].add(v)
                    g[v].add(u)
    t = quickbb(g)

    # The root bag is the one that contains all the externals
    ext = set(rhs.ext())
    for bag in t:
        if ext.issubset(bag.nodes):
            root = bag
            break
    else:
        assert False

    newrules = []
    i = 0
    
    def visit(bag, parent):
        nonlocal i

        # lhs and external nodes
        if parent is None:
            ext = rule.rhs().ext()
            lhs = rule.lhs()
        else:
            ext = list(bag.nodes & parent.nodes)
            lhs = fggs.EdgeLabel(f'{rule.rule_id()}_{i}',
                                 is_terminal=False,
                                 node_labels=[v.label() for v in ext])
            i += 1
        
        # rhs
        rhs = fggs.FactorGraph()
        for v in bag.nodes:
            rhs.add_node(v)
        rhs.set_ext(ext)

        # terminal edges
        for e in rule.rhs().edges():
            if (bag.nodes.issuperset(e.nodes()) and
                (parent is None or not parent.nodes.issuperset(e.nodes()))):
                rhs.add_edge(e)

        # nonterminal edges and recurse on children
        for n in t[bag]:
            if n != parent:
                child_lhs, child_ext = visit(n, bag)
                rhs.add_edge(fggs.Edge(child_lhs, child_ext))

        newrule = fggs.FGGRule(lhs, rhs)
        newrules.append(newrule)
        return (lhs, ext)
    
    visit(root, None)
    
    return newrules

def factorize(g):
    """Factorize a FGG's rules into smaller rules, hopefully with
    lower maximum treewidth.
    """
    gnew = fggs.FGGRepresentation()
    for r in g.all_rules():
        for rnew in factorize_rule(r):
            gnew.add_rule(rnew)
    gnew.set_start_symbol(g.start_symbol())
    return gnew

if __name__ == "__main__":
    import json
    import sys
    import formats

    if len(sys.argv) != 3:
        print('usage: factorize.py <infile> <outfile>', file=sys.stderr)
        exit(1)

    g = formats.json_to_fgg(json.load(open(sys.argv[1])))
    g = factorize(g)
    json.dump(formats.fgg_to_json(g), open(sys.argv[2], 'w'), indent=2)
