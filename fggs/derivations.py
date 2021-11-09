__all__ = ['start_graph', 'replace_edges']

from typing import Dict

from fggs.fggs import *

def start_graph(g: HRG) -> Graph:
    """Construct a graph consisting of a single Edge labeled by the start
    nonterminal symbol."""
    ret = Graph()
    s = g._start
    e = Edge(s, [Node(l) for l in s.type()])
    ret.add_edge(e)
    return ret
        
def replace_edges(graph: Graph, replacements: Dict[Edge, Graph]):

    """Return a copy of `graph` in which, for every item `(edge,
    repl)` in `replacements`, `edge` (which must be labeled by a
    `Nonterminal`) is replaced with a copy of graph `repl`
    (which must have the same type as `edge`).
    """

    for (edge, repl) in replacements.items():
        if not edge.label.is_nonterminal:
            raise ValueError("Only a nonterminal-labeled edge can be replaced.")

        if isinstance(repl, HRGRule):
            if edge.label != repl.lhs:
                raise ValueError("An edge can only be replaced with a HRGRule with a matching left-hand side.")
        elif isinstance(repl, Graph):
            if edge.label.type() != repl.type():
                raise ValueError("A graph fragment can only replace a edge with the same type.")
        else:
            raise TypeError("The replacement for an edge must be a HRGRule or a Graph.")


    ret = Graph()
    for v in graph.nodes():
        ret.add_node(v)
    ret.ext = graph.ext
    for e in graph.edges():
        if e not in replacements:
            ret.add_edge(e)
    for e in replacements:
        repl = replacements[e]
        if isinstance(repl, HRGRule):
            repl = repl.rhs
        rnodes = {}
        for ve, vr in zip(e.nodes, repl.ext):
            rnodes[vr] = ve
        for v in repl.nodes():
            if v not in rnodes: # i.e., if v not in repl.ext
                if v.id in ret._node_ids:
                    vcopy = Node(v.label) # generate fresh id
                else:
                    vcopy = v
                rnodes[v] = vcopy
                ret.add_node(vcopy)
        for er in repl.edges():
            if er.id in ret._edge_ids:
                er = Edge(er.label, [rnodes[v] for v in er.nodes]) # fresh id
            else:
                er = Edge(er.label, [rnodes[v] for v in er.nodes], id=er.id)
            ret.add_edge(er)
    return ret


