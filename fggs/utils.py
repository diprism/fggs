from typing import Iterable, Union
from fggs.fggs import *


def unique_label_name(name: str, labs: Iterable[Union[NodeLabel, EdgeLabel]]) -> str:
    """Given an name, modify it until it does not overlap with
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
    edge_labels = [edge.label for edge in graph.edges()]
    start_name = unique_label_name("<S>", edge_labels)

    start   = EdgeLabel(start_name, graph.type, is_nonterminal=True)
    rule    = HRGRule(start, graph)
    grammar = HRG(start)
    grammar.add_rule(rule)
    return grammar    

def singleton_fgg(fac_graph: FactorGraph) -> FGG:
    """Return an FGG which generates just one factor graph, `fac_graph`."""
    return FGG(singleton_hrg(fac_graph.graph), fac_graph.interp)
