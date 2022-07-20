__all__ = ['FGGDerivation', 'start_graph', 'replace_edge']

from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass
from fggs.fggs import *

@dataclass
class FGGDerivation:
    fgg: FGG
    rule: HRGRule
    asst: Dict[Node, Any]
    children: Dict[Edge, 'FGGDerivation']

    def derive(self):
        """Returns the factor graph and assignment derived by this derivation."""

        graph = FactorGraph()
        edge = Edge(self.rule.lhs, [Node(l) for l in self.rule.lhs.type])
        graph.add_edge(edge)
        asst: Dict[Node, Any] = {}
        
        def visit(deriv: FGGDerivation, edge: Edge):
            (node_map, edge_map) = replace_edge(graph, edge, deriv.rule.rhs)
            for node in deriv.rule.rhs.nodes():
                asst[node_map[node]] = deriv.asst[node]
            for child in deriv.children:
                visit(deriv.children[child], edge_map[child])

        visit(self, edge)
        graph.domains = self.fgg.domains
        graph.factors = self.fgg.factors
        return (graph, asst)
    

def start_graph(g: HRG) -> Graph:
    """Construct a graph consisting of a single Edge labeled by the start
    nonterminal symbol."""
    ret = Graph()
    s = g.start
    e = Edge(s, [Node(l) for l in s.type])
    ret.add_edge(e)
    return ret
        

def replace_edge(graph: Graph, edge: Edge, replacement: Graph) -> Tuple[Dict[Node, Node], Dict[Edge, Edge]]:
    """Destructively replace an edge with a graph.

    - `graph`: A Graph.
    - `edge`: An Edge in `graph` labeled by a nonterminal.
    - `replacement`: A Graph with the same type as `edge`.

    Returns: a pair (node_map, edge_map), where
    - `node_map` maps nodes in `replacement` to nodes in the result.
    - `edge_map` maps edges in `replacement` to edges in the result.
    """
    if edge.label.type != replacement.type:
        raise ValueError('An edge can only be replaced with a graph having the same type')
    graph.remove_edge(edge)

    # Copy replacement into graph, assigning fresh ids to
    # replacement's nodes and edges.
    
    node_map: Dict[Node, Node] = {}
    for gnode, rnode in zip(edge.nodes, replacement.ext):
        node_map[rnode] = gnode
    for rnode in replacement.nodes():
        if rnode not in node_map: # i.e., if rnode is not external
            gnode = Node(rnode.label)
            node_map[rnode] = gnode
            graph.add_node(gnode)
            
    edge_map: Dict[Edge, Edge] = {}
    for redge in replacement.edges():
        gnodes = tuple(node_map[rnode] for rnode in redge.nodes)
        gedge = Edge(redge.label, gnodes)
        edge_map[redge] = gedge
        graph.add_edge(gedge)

    return (node_map, edge_map)
