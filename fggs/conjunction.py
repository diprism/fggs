__all__ = ['conjoin_hrgs']

from fggs import fggs
from fggs import utils

def nonterminal_pairs(hrg1, hrg2):
    """Generate new nonterminal symbols for all possible pairs of nonterminals from hrg1 and hrg2.
    Returns a dict mapping from pairs of nonterminals to new nonterminals.

    Checks for two types of collision:
    - Between two pairs of nonterminals, e.g., "X" + "Y,Z" and "X,Y" + "Z" both become "<X,Y,Z>"
    - Between nonterminals and terminals, e.g., terminal "<X,Y>" collides with the pairing of nonterminals "X" and "Y"
    """

    nt_map = {}
    labels = set(hrg1.edge_labels()) | set(hrg2.edge_labels())
    for el1 in hrg1.nonterminals():
        for el2 in hrg2.nonterminals():
            new_nt_name = f'<{el1.name},{el2.name}>'
            new_nt_name = utils.unique_label_name(new_nt_name, labels)
            new_nt = fggs.EdgeLabel(new_nt_name,
                                    node_labels=el1.type,
                                    is_nonterminal=True)
            nt_map[el1,el2] = new_nt
            labels.add(new_nt)
    return nt_map

def check_namespace_collisions(hrg1, hrg2):
    """Checks whether two HRGs have any conflicting NodeLabels or EdgeLabels."""
    # Check for conflicting NodeLabels
    # (Currently, it's actually not possible for NodeLabels to conflict, but in the future, it might be.)
    node_collisions = []
    for nl1 in hrg1.node_labels():
        if hrg2.has_node_label_name(nl1.name):
            nl2 = hrg2.get_node_label(nl1.name)
            if nl1 != nl2:
                node_collisions.append((nl1, nl2))
    # Check for conflicting EdgeLabels
    edge_collisions = []
    for el1 in hrg1.edge_labels():
        if hrg2.has_edge_label_name(el1.name):
            el2 = hrg2.get_edge_label(el1.name)
            if el1 != el2:
                edge_collisions.append((el1, el2))
    return (node_collisions, edge_collisions)

def conjoinable(rule1, rule2):
    """Test whether two HRG rules are conjoinable."""
    
    # Must have same Nodes (in terms of Node id) with same NodeLabels
    if set(rule1.rhs.nodes()) != set(rule2.rhs.nodes()):
        return False
    # Must have same nonterminal Edges (in terms of Edge id)
    # which can have different EdgeLabels, but must connect to same Nodes
    nts1 = set([(edge.id, tuple([node.id for node in edge.nodes]))
                for edge in rule1.rhs.edges()
                if edge.label.is_nonterminal])
    nts2 = set([(edge.id, tuple([node.id for node in edge.nodes]))
                for edge in rule2.rhs.edges()
                if edge.label.is_nonterminal])
    if nts1 != nts2:
        return False
    # Must have same external nodes
    ext1 = [node.id for node in rule1.rhs.ext]
    ext2 = [node.id for node in rule2.rhs.ext]
    if ext1 != ext2:
        return False
    return True

def conjoin_rules(rule1, rule2, nt_map):
    """Conjoin two HRG rules.

    Assumes rules are conjoinable.
    Does not check for conjoinability."""
    
    new_lhs = nt_map[rule1.lhs, rule2.lhs]
    new_rhs = fggs.Graph()
    # add nodes
    for node in rule1.rhs.nodes():
        new_rhs.add_node(node)
    # set external nodes
    new_rhs.ext = rule1.rhs.ext
    # add nonterminal edges
    nts1 = sorted([edge for edge in rule1.rhs.edges() if edge.label.is_nonterminal],
                  key=lambda edge: edge.id)
    nts2 = sorted([edge for edge in rule2.rhs.edges() if edge.label.is_nonterminal],
                  key=lambda edge: edge.id)
    for (edge1,edge2) in zip(nts1,nts2):
        new_rhs.add_edge(fggs.Edge(label=nt_map[edge1.label, edge2.label],
                                   nodes=edge1.nodes,
                                   id=edge1.id))
    # add terminal edges
    ts1 = [e for e in rule1.rhs.edges() if e.label.is_terminal]
    ts2 = [e for e in rule2.rhs.edges() if e.label.is_terminal]
    for edge in ts1 + ts2:
        new_rhs.add_edge(edge)
    return fggs.HRGRule(lhs=new_lhs, rhs=new_rhs)

def conjoin_hrgs(hrg1, hrg2):
    """Conjoin two HRGS."""
    (n_col, e_col) = check_namespace_collisions(hrg1, hrg2)
    for (nl1, nl2) in n_col:
        raise ValueError(f"Cannot conjoin hrg1 and hrg2 because they each have a different NodeLabel called {nl1.name}")
    for (el1, el2) in e_col:
        if el1.is_terminal and el2.is_terminal:
            raise ValueError(f"Cannot conjoin hrg1 and hrg2 because they each have a different terminal EdgeLabel called {el1.name}")
    nt_map = nonterminal_pairs(hrg1, hrg2)
    new_hrg = fggs.HRG(nt_map[hrg1.start, hrg2.start])
    # add rules
    for rule1 in hrg1.all_rules():
        for rule2 in hrg2.all_rules():
            if conjoinable(rule1, rule2):
                new_hrg.add_rule(conjoin_rules(rule1, rule2, nt_map))
    return new_hrg
