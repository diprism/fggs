from typing import Optional
from fggs.fggs import HRG, HRGRule, Node, Edge, EdgeLabel, Graph

def adjoint_hrg(g: HRG, top=None):
    """Form the adjoint of an HRG, used for computing gradients and 
    expectations.

    - g: The HRG to form the adjoint of.
    - top: Add rules X -> top[X]

    Returns: a pair (gbar, index), where
    - gbar: The adjoint HRG
    - index: dict mapping from terminal and nonterminal edge labels to their adjoint edge labels
    """
    if top is None: top = {}
    
    gbar = HRG()

    bar = {}
    for x in g.edge_labels():
        xbar = x.name + "_bar"
        i = 1
        while g.has_edge_label(xbar):
            xbar = x.name + f"_bar{i}"
            i += 1
        bar[x] = EdgeLabel(xbar, x.node_labels, is_nonterminal=True)

    # The adjoint HRG doesn't really need a start symbol,
    # but some code doesn't like HRGs without start symbols,
    # so choose one.
    s = g.start_symbol
    gbar.start_symbol = bar[s]

    for x in top:
        rhs = Graph()
        nodes = [Node(lab) for lab in x.node_labels]
        rhs.set_ext(nodes)
        rhs.add_edge(Edge(top[x], nodes))
        gbar.add_rule(HRGRule(bar[x], rhs))

    for p in g.all_rules():
        gbar.add_rule(p)
        for e in p.rhs.edges():
            rbar = p.rhs.copy()
            rbar.remove_edge(e)
            rbar.add_edge(Edge(bar[p.lhs], rbar.ext()))
            rbar.set_ext(e.nodes)
            gbar.add_rule(HRGRule(bar[e.label], rbar))
            
    return gbar, bar
