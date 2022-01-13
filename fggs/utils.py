from fggs.fggs import *

def singleton_hrg(graph: Graph) -> HRG:
    """Return an HRG which generates just one graph, `graph`."""

    # Construct a new edge label which is not already used in the graph
    edge_labels = [edge.label for edge in graph.edges()]
    edge_label_names = [label.name for label in edge_labels]
    start_name = "<S>"
    while start_name in edge_label_names:
        start_name = "<" + start_name + ">"

    start   = EdgeLabel(start_name, graph.type, is_nonterminal=True)
    rule    = HRGRule(start, graph)
    grammar = HRG(start)
    grammar.add_rule(rule)
    return grammar
    

def singleton_fgg(fac_graph: FactorGraph) -> FGG:
    """Return an FGG which generates just one graph, `graph`."""
    return FGG(singleton_HRG(fac_graph.graph), fac_graph.interp)
