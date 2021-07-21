__all__ = ['conjoin_fggs']

import warnings
from fggs import fggs


def check_namespace_collisions(fgg1, fgg2):
    """Checks whether two FGGs have any conflicting NodeLabels or EdgeLabels."""
    # check for conflicting NodeLabels
    node_collisions = []
    for nl1 in fgg1.node_labels():
        if fgg2.has_node_label(nl1.name):
            nl2 = fgg2.get_node_label(nl1.name)
            if nl1 != nl2:
                node_collisions.append((nl1, nl2))
    # check for conflicting EdgeLabels
    edge_collisions = []
    for el1 in fgg1.edge_labels():
        if fgg2.has_edge_label(el1.name):
            el2 = fgg2.get_edge_label(el1.name)
            if el1 != el2:
                edge_collisions.append((el1, el2))
    return (node_collisions, edge_collisions)

def conjoinable(rule1, rule2):
    """Test whether two FGG rules are conjoinable."""
    
    # Must have same Nodes (in terms of Node id) with same NodeLabels
    if rule1.rhs().nodes() != rule2.rhs().nodes():
        return False
    # Must have same nonterminal Edges (in terms of Edge id)
    # which can have different EdgeLabels, but must connect to same Nodes
    nts1 = set([(edge.id, tuple([node.id for node in edge.nodes]))
                for edge in rule1.rhs().nonterminals()])
    nts2 = set([(edge.id, tuple([node.id for node in edge.nodes]))
                for edge in rule2.rhs().nonterminals()])
    if nts1 != nts2:
        return False
    # Must have same external nodes
    ext1 = [node.id for node in rule1.rhs().ext()]
    ext2 = [node.id for node in rule2.rhs().ext()]
    if ext1 != ext2:
        return False
    return True

def conjoin_rules(rule1, rule2):
    """Conjoin two FGG rules.

    Assumes rules are conjoinable.
    Does not check for conjoinability."""
    
    new_lhs = fggs.EdgeLabel(name=f"<{rule1.lhs().name},{rule2.lhs().name}>",
                             is_terminal=False,
                             node_labels=rule1.lhs().type())
    new_rhs = fggs.FactorGraph()
    # add nodes
    for node in rule1.rhs().nodes():
        new_rhs.add_node(node)
    # set external nodes
    new_rhs.set_ext(rule1.rhs().ext())
    # add nonterminal edges
    nts1 = sorted([edge for edge in rule1.rhs().nonterminals()],
                  key=lambda edge: edge.id)
    nts2 = sorted([edge for edge in rule2.rhs().nonterminals()],
                  key=lambda edge: edge.id)
    new_labels = dict()
    for (edge1,edge2) in zip(nts1,nts2):
        name = f"<{edge1.label.name},{edge2.label.name}>"
        if name in new_labels:
            label = new_labels[name]
        else:
            label = fggs.EdgeLabel(name=name,
                                   is_terminal=False,
                                   node_labels=edge1.label.type())
            new_labels[name] = label
        new_rhs.add_edge(fggs.Edge(label=label, nodes=edge1.nodes, id=edge1.id))
    # add terminal edges
    ts1 = rule1.rhs().terminals()
    ts2 = rule2.rhs().terminals()
    for edge in ts1 + ts2:
        new_rhs.add_edge(edge)
    return fggs.FGGRule(lhs=new_lhs, rhs=new_rhs)

def conjoin_fggs(fgg1, fgg2):
    """Conjoin two FGGS."""
    # first check for namespace collisions, and warn the user
    (n_col, e_col) = check_namespace_collisions(fgg1, fgg2)
    for (nl1, nl2) in n_col:
        warnings.warn(f"Warning during conjunction: fgg1 and fgg2 each have a different NodeLabel called {nl1.name}")
    for (el1, el2) in e_col:
        warnings.warn(f"Warning during conjunction: fgg1 and fgg2 each have a different EdgeLabel called {el1.name}")
    new_fgg = fggs.FGG()
    # add rules
    for rule1 in fgg1.all_rules():
        for rule2 in fgg2.all_rules():
            if conjoinable(rule1, rule2):
                new_fgg.add_rule(conjoin_rules(rule1, rule2))
    # set the start symbol
    # (may not actually be used in any rules)
    start_name = f"<{fgg1.start_symbol().name},{fgg2.start_symbol().name}>"
    if new_fgg.has_edge_label(start_name):
        start_label = new_fgg.get_edge_label(start_name)
    else:
        start_label = fggs.EdgeLabel(name=start_name,\
                                     is_terminal=False,\
                                     node_labels=fgg1.start_symbol().type())
    new_fgg.set_start_symbol(start_label)
    return new_fgg
