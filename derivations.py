from typing import Dict

from fgg_representation import FGGRepresentation, FGGRule, FactorGraph, Edge, Node

def start_graph(g: FGGRepresentation) -> FactorGraph:
    """Construct a graph consisting of a single Edge labeled by the start
    nonterminal symbol."""
    ret = FactorGraph()
    s = g._start
    e = Edge(s, [Node(l) for l in s.type()])
    ret.add_edge(e)
    return ret
        
def replace_edges(graph: FactorGraph, replacements: Dict[Edge, FactorGraph]):

    """Return a copy of `graph` in which, for every item `(edge,
    repl)` in `replacements`, `edge` (which must be labeled by a
    `Nonterminal`) is replaced with a copy of graph `repl`
    (which must have the same type as `edge`).
    """

    for (edge, repl) in replacements.items():
        if not edge.label().is_nonterminal:
            raise ValueError("Only a nonterminal-labeled edge can be replaced.")

        if isinstance(repl, FGGRule):
            if edge.label() != repl.lhs():
                raise ValueError("An edge can only be replaced with an FGGRule with a matching left-hand side.")
        elif not isinstance(repl, FactorGraph):
            raise TypeError("The replacement for an edge must be an FGGRule or a FactorGraph.")

        if edge.label().type() != repl.type():
            raise ValueError("A graph fragment can only replace a edge with the same type.")

    ret = FactorGraph()
    nodes = {}
    for v in graph.nodes():
        v = v.copy()
        nodes[v.id()] = v
        ret.add_node(v)
    for e in graph.edges():
        if e in replacements:
            repl = replacements[e]
            if isinstance(repl, FGGRule):
                repl = repl.rhs()
            rnodes = {}
            for ve, vr in zip(e.nodes(), repl.ext()):
                rnodes[vr.id()] = nodes[ve.id()]
            for v in repl.nodes():
                if v.id() not in rnodes:
                    vcopy = Node(v.label()) # don't keep id
                    rnodes[v.id()] = vcopy
                    ret.add_node(vcopy)
            for e in repl.edges():
                e = Edge(e.label(), [rnodes[v.id()] for v in e.nodes()])
                ret.add_edge(e)
        else:
            e = Edge(e.label(), [nodes[v.id()] for v in e.nodes()])
            ret.add_edge(e)
    return ret


