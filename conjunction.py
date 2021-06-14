import fgg_representation as fggs

# Test whether two FGG rules are conjoinable.
def conjoinable(rule1, rule2):
    # Must have same Nodes (in terms of Node id) with same NodeLabels
    nodes1 = set([(node.id(), node.label().name()) for node in rule1.rhs().nodes()])
    nodes2 = set([(node.id(), node.label().name()) for node in rule2.rhs().nodes()])
    if nodes1 != nodes2:
        return False
    # Must have same nonterminal Edges (in terms of Edge id)
    # which can have different EdgeLabels, but must connect to same Nodes
    nts1 = set([(edge.id(), tuple([node.id() for node in edge.nodes()]))\
                for edge in rule1.rhs().nonterminals()])
    nts2 = set([(edge.id(), tuple([node.id() for node in edge.nodes()]))\
                for edge in rule2.rhs().nonterminals()])
    if nts1 != nts2:
        return False
    # Must have same external nodes
    ext1 = [node.id() for node in rule1.rhs().ext()]
    ext2 = [node.id() for node in rule2.rhs().ext()]
    if ext1 != ext2:
        return False
    return True

# Assumes rules are conjoinable.
# Does not check for conjoinability.
def conjoin_rules(rule1, rule2):
    new_lhs = fggs.EdgeLabel(name=f"<{rule1.lhs().name()},{rule2.lhs().name()}>",\
                             is_terminal=False,\
                             type=rule1.lhs().type())
    new_rhs = fggs.FactorGraph()
    # add nodes
    new_nodes = dict()
    for node in rule1.rhs().nodes():
        new_node = fggs.Node(node.label(), node.id())
        new_rhs.add_node(new_node)
        new_nodes[node.id()] = new_node
    # set external nodes
    new_ext = [new_nodes[node.id()] for node in rule1.rhs().ext()]
    new_rhs.set_ext(new_ext)
    # add nonterminal edges
    nts1 = sorted([edge for edge in rule1.rhs().nonterminals()],\
                  key=lambda edge: edge.id())
    nts2 = sorted([edge for edge in rule2.rhs().nonterminals()],\
                  key=lambda edge: edge.id())
    new_labels = dict()
    for (edge1,edge2) in zip(nts1,nts2):
        name = f"<{edge1.label().name()},{edge2.label().name}>"
        if name in new_labels:
            label = new_labels[name]
        else:
            label = fggs.EdgeLabel(name=name,\
                                   is_terminal=False,\
                                   type=edge1.label().type())
            new_labels[name] = label
        new_edge = fggs.Edge(label=label,\
                             nodes=[new_nodes[n.id()] for n in edge1.nodes()],\
                             id=edge1.id())
        new_rhs.add_edge(new_edge)
    # add terminal edges
    ts1 = rule1.rhs().terminals()
    ts2 = rule2.rhs().terminals()
    for edge in ts1 + ts2:
        new_edge = fggs.Edge(label=edge.label,\
                             nodes=[new_nodes[n.id()] for n in edge.nodes()],\
                             id=edge.id())
        new_rhs.add_edge(new_edge)
    return fggs.FGGRule(lhs=new_lhs, rhs=new_rhs)

# Conjoin two FGGs.
def conjoin_fggs(fgg1, fgg2):
    new_fgg = fggs.FGGRepresentation()
    # add rules
    for rule1 in fgg1.all_rules():
        for rule2 in fgg2.all_rules():
            if conjoinable(rule1, rule2):
                new_fgg.add_rule(conjoin_rules(rule1, rule2))
    # if the appropriate start symbol exists, set it
    # otherwise leave it blank (and maybe consider this a conjunction fail?)
    start_name = f"<{fgg1.start_symbol().name()},{fgg2.start_symbol().name()}>"
    if new_fgg.has_edge_label(start_name):
        start_label = new_fgg.get_edge_label(start_name)
        new_fgg.set_start_symbol(start_label)
    return new_fgg
