from fgg_representation import FGGRepresentation, FGGRule, FactorGraph, Edge, Node

def start_graph(g: FGGRepresentation) -> FactorGraph:
    """Construct a graph consisting of a single Edge labeled by the start
    nonterminal symbol."""
    ret = FactorGraph()
    s = g._start
    e = Edge(s, [Node(l) for l in s.type()])
    ret.add_edge(e)
    return ret
        
def replace_edge(graph: FactorGraph, edge: Edge, other):
    """Replace `edge` (which must be labeled by a `Nonterminal`) with
    a copy of graph `other` (which must have the same type as `edge`)."""

    if graph is other:
        raise ValueError("A FactorGraph's edge can't be replaced with the FactorGraph itself.")

    if not edge.label().is_nonterminal():
        raise ValueError("Only a nonterminal-labeled edge can be replaced.")
    
    if isinstance(other, FGGRule):
        if edge.label() != other.lhs():
            raise ValueError("An edge can only be replaced with an FGGRule with a matching left-hand side.")
        other = other.rhs()
    if not isinstance(other, FactorGraph):
        raise TypeError("The replacement for an edge must be an FGGRule or a FactorGraph.")

    if edge.label().type() != other.type():
        raise ValueError("A graph fragment can only replace a edge with the same type.")

    # We can keep edge's attachment nodes or other's external nodes.
    # Arbitrarily keep the former.
    graph.remove_edge(edge)
    node_map = dict(zip(other.ext(), edge.nodes()))
    for v in other.nodes():
        if v not in other.ext():
            if v.id() in graph._node_ids:
                v1 = Node(v.label())
                node_map[v] = v1
                v = v1
            graph.add_node(v)
    for e in other.edges():
        if e.id() in graph._edge_ids or any(v in node_map for v in e.nodes()):
            e = Edge(e.label(), [node_map.get(v, v) for v in e.nodes()])
        graph.add_edge(e)

