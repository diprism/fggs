from typing import Iterable, Union, List, Set, Dict, TypeVar, Tuple
from fggs.fggs import *
import itertools

def unique_label_name(name: str, labs: Iterable[Union[NodeLabel, EdgeLabel]]) -> str:
    """Given a name, modify it until it does not overlap with
    a given set of NodeLabel/EdgeLabel names."""
    names = [lab.name for lab in labs]
    new_name = name
    i = 1
    while new_name in names:
        new_name = f'{name}_{i}'
        i += 1
    return new_name


def singleton_hrg(graph: Graph) -> HRG:
    """Return an HRG which generates just one graph, `graph`."""

    # Construct a new edge label name which is not already used in the graph
    edge_labels = graph.edge_labels()
    start_name = unique_label_name("<S>", edge_labels)

    start   = EdgeLabel(start_name, graph.type, is_nonterminal=True)
    rule    = HRGRule(start, graph)
    grammar = HRG(start)
    grammar.add_rule(rule)
    return grammar    

def singleton_fgg(fac_graph: FactorGraph) -> FGG:
    """Return an FGG which generates just one factor graph, `fac_graph`."""
    fgg = FGG.from_hrg(singleton_hrg(fac_graph))
    fgg.domains = fac_graph.domains
    fgg.factors = fac_graph.factors
    return fgg


def nonterminal_graph(hrg: HRG) -> Dict[EdgeLabel, Set[EdgeLabel]]:
    """Returns a directed graph g (of type Dict[EdgeLabel,
    Set[EdgeLabel]]) such that g[x] contains y iff there is rule with
    lhs x and a nonterminal y occurring on the rhs.
    """
    g: Dict[EdgeLabel, Set[EdgeLabel]] = {x:set() for x in hrg.nonterminals()}
    for r in hrg.all_rules():
        for e in r.rhs.edges():
            if e.label.is_nonterminal:
                g[r.lhs].add(e.label)
    return g

T = TypeVar('T')
def scc(g: Dict[T, Set[T]]) -> List[Set[T]]:
    """Decompose an HRG into a its strongly-connected components using
    Tarjan's algorithm.

    Returns a list of sets of nonterminal EdgeLabels. The list is in
    topological order: there is no rule with a lhs in an earlier
    component and an rhs nonterminal in a later component.

    Robert Tarjan. Depth-first search and linear graph
    algorithms. SIAM J. Comput., 1(2),
    146-160. https://doi.org/10.1137/0201010

    Based on pseudocode from https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm

    """
    
    index = 0
    indexof = {}    # order of nodes in DFS traversal
    lowlink = {}    # lowlink[v] = min(indexof[w] | w is v or a descendant of v)
    stack = []      # path from start node to current node
    onstack = set() # = set(stack)
    comps = []

    def visit(v):
        nonlocal index
        indexof[v] = lowlink[v] = index
        index += 1
        stack.append(v)
        onstack.add(v)

        for w in g[v]:
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
    
    for v in g:
        if v not in indexof:
            visit(v)

    return comps


def naive_graph_isomorphism(g1: Graph, g2: Graph):
    """Find an isomorphism from g1 to g2, if any. Used in unit tests only.

    If g1 and g2 are isomorphic, return (True, f) where f is a mapping
    from nodes of g1 to nodes of g2. Else, return (False, msg) where
    msg is a string explaining why the graphs are not isomorphic.
    """
    nodes1 = set(node.label for node in g1.nodes())
    nodes2 = set(node.label for node in g2.nodes())
    if nodes1 != nodes2:
        return (False, f'different node labels ({nodes1} != {nodes2}')
    edges1 = set(edge.label for edge in g1.edges())
    edges2 = set(edge.label for edge in g2.edges())
    if edges1 != edges2:
        return (False, f'different edge labels ({edges1} != {edges2})')
    if g1.type != g2.type:
        return (False, f'different graph types ({g1.type} != {g2.type})')
    
    order1 = list(g1.nodes())
    for order2 in itertools.permutations(g2.nodes()):
        if [node.label for node in order1] != [node.label for node in order2]:
            continue
        map1 = {node:i for (i, node) in enumerate(order1)}
        map2 = {node:i for (i, node) in enumerate(order2)}
        edges1 = set((edge.label, tuple(map1[node] for node in edge.nodes)) for edge in g1.edges())
        edges2 = set((edge.label, tuple(map2[node] for node in edge.nodes)) for edge in g2.edges())
        if edges1 != edges2: continue
        if [map1[node] for node in g1.ext] != [map2[node] for node in g2.ext]:
            continue
        return (True, dict(zip(order1, order2)))
    return(False, 'graphs are not isomorphic')

