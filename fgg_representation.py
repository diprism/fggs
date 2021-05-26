# TODO: improve the comment structure
# TODO: change how I number nodes/edges to make pretty-printing work better
# TODO: add custom sorting functions for nodes/edges based on their IDs?

from typing import Optional, Iterable
from domains import Domain
from factors import Factor



class NodeLabel:

    def __init__(self, name: str, domain: Domain):
        self._name   = name
        self._domain = domain
    
    def name(self):
        return self._name
    
    def domain(self):
        return self._domain
    
    def __str__(self):
        return f"NodeLabel {self._name} with Domain {self._domain}"



class EdgeLabel:
    
    def __init__(self, name: str, is_terminal: bool, node_labels: Iterable[NodeLabel], fac: Optional[Factor] = None):
        self._name        = name
        self._is_terminal = is_terminal
        self._node_labels = tuple(node_labels)
        self.set_factor(fac)

    def name(self):
        return self._name
    
    def is_terminal(self):
        return self._is_terminal
        
    def is_nonterminal(self):
        return not self._is_terminal
        
    def arity(self):
        return len(self._node_labels)
        
    def type(self):
        return self._node_labels
    
    def factor(self):
        return self._factor
    
    def set_factor(self, fac: Optional[Factor]):
        if self._is_terminal:
            if fac is None:
                raise ValueError(f"Terminal edge label {self._name} is missing a factor.")
            elif fac.domains() != tuple([nl.domain() for nl in self.type()]):
                raise ValueError(f"Factor function has the wrong type for edge label {self._name}.")
        if not self._is_terminal:
            if fac is not None:
                raise ValueError(f"Nonterminal edge label {self._name} should not have a factor.")
        self._factor = fac

    def __str__(self):
        return self.to_string(0)
    def to_string(self, indent):
        string = "\t"*indent
        if self.is_terminal():
            string += f"Terminal EdgeLabel {self._name} with arity {self.arity()} and Factor {self._factor}"
        else:
            string += f"Nonterminal EdgeLabel {self._name} with arity {self.arity()}"
        if self.arity() != 0:
            string += " and endpoints of type:"
            for i, node_label in enumerate(self.type()):
                string += "\n\t" + "\t"*indent + f"{i+1}. NodeLabel {node_label.name()}"
        return string

    

class Node:
    
    node_count = 0
    
    def __init__(self, label: NodeLabel):
        Node.node_count += 1
        self._id     = Node.node_count
        self._label  = label
        self._value  = None
    
    def node_id(self):
        return self._id
    
    def label(self):
        return self._label
    
    def has_value(self):
        return self._value != None
    
    def value(self):
        return self._value
    
    def set_value(self, val):
        if (val != None) and (not self._label.domain().contains(val)):
            raise Exception(f"Node with label {self._label.name()} cannot take value {val}.")
        self._value = val
    
    def unset_value(self):
        self._value = None
        
    def __str__(self):
        return f"Node {self._id} with NodeLabel {self.label().name()} and value {self.value()}"



class Edge:

    edge_count = 0
    
    def __init__(self, label: EdgeLabel, nodes: Iterable[Node]):
        Edge.edge_count += 1
        self._id     = Edge.edge_count
        if label.type() != tuple([node.label() for node in nodes]):
            raise Exception(f"Can't use edge label {label.name()} with this set of nodes.")
        self._label = label
        self._nodes = tuple(nodes)

    def edge_id(self):
        return self._id
    
    def label(self):
        return self._label

    def nodes(self):
        return self._nodes
    
    def node_at(self, i):
        return self._nodes[i]
    
    def apply_factor(self):
        if not self._label.is_terminal():
            raise Exception(f"Nonterminal edge {label.name()} cannot apply factor.")
        if any([not node.has_value() for node in self._nodes]):
            raise Exception(f"Cannot apply factor to nodes which have no value.")
        return self._label.factor().apply([node.value() for node in self._nodes])
    
    def __str__(self):
        return self.to_string(0, True)
    def to_string(self, indent, verbose):
        arity = len(self.nodes())
        string = "\t"*indent
        string += f"Edge {self._id} with EdgeLabel {self.label().name()}, connecting to {arity} nodes"
        if arity > 0:
            string += ":"
            for node in self._nodes:
                string += "\n\t" + "\t"*indent
                if verbose:
                    string += f"{node}"
                else:
                    string += f"Node {node.node_id()}"
        return string



class FactorGraph:

    def __init__(self):
        self._nodes = set()
        self._edges = set()
        self._ext   = tuple()
    
    def nodes(self):
        return list(self._nodes)
    
    def edges(self):
        return list(self._edges)
    
    def ext(self):
        return self._ext
    
    def arity(self):
        return len(self._ext)
    
    def type(self):
        return tuple([node.label() for node in self._ext])
    
    def add_node(self, node: Node):
        self._nodes.add(node)

    def add_edge(self, edge: Edge):
        for node in edge.nodes():
            if node not in self._nodes:
                self._nodes.add(node)
        self._edges.add(edge)

    def set_ext(self, nodes: Iterable[Node]):
        for node in nodes:
            if node not in self._nodes:
                self._nodes.add(node)
        self._ext = tuple(nodes)
    
    def __str__(self):
        return self.to_string(0)
    def to_string(self, indent):
        num_nodes = len(self._nodes)
        num_edges = len(self._edges)
        string = "\t"*indent + f"Factor graph with {num_nodes} nodes and {num_edges} edges"
        if num_nodes > 0:
            string += "\n" + "\t"*indent + "Nodes:"
            for node in self._nodes:
                string += "\n\t" + "\t"*indent + f"{node}"
        if num_edges > 0:
            string += "\n" + "\t"*indent + "Edges:"
            for edge in self._edges:
                string += "\n" + edge.to_string(indent+1, False)
        return string



class FGGRule:

    rule_count = 0
    
    def __init__(self, lhs: EdgeLabel, rhs: FactorGraph):
        if lhs.is_terminal():
            raise Exception(f"Can't make FGG rule with terminal left-hand side.")
        if (lhs.type() != rhs.type()):
            raise Exception(f"Can't make FGG rule: left-hand side of type ({','.join(l.name() for l in lhs.type())}) not compatible with right-hand side of type ({','.join(l.name() for l in rhs.type())}).")
        FGGRule.rule_count += 1
        self._id  = FGGRule.rule_count
        self._lhs = lhs
        self._rhs = rhs

    def rule_id(self):
        return self._id
    
    def lhs(self):
        return self._lhs
    
    def rhs(self):
        return self._rhs
    
    def __str__(self):
        return self.to_string(0)
    def to_string(self, indent):
        string = "\t"*indent
        string += f"FGGRule with left-hand side {self._lhs.name()} and right-hand side as follows:\n"
        string += self._rhs.to_string(indent+1)
        return string



class FGGRepresentation:
    
    def __init__(self):
        self._node_labels  = dict()    # map from names to NodeLabels    
        self._nonterminals = dict()    # map from names to EdgeLabels
        self._terminals    = dict()    # map from names to EdgeLabels
        self._start        = None      # start symbol, an EdgeLabel which has arity 0
        self._rules        = dict()    # one set of rules for each nonterminal edge label

    def add_node_label(self, label: NodeLabel):
        name = label.name()
        if name in self._node_labels:
            if self._node_labels[name] != label:
                raise Exception(f"There is already a node label with name {name}.")
        self._node_labels[label.name()] = label

    def get_node_label(self, name):
        return self._node_labels[name]

    def node_labels(self):
        return [self._node_labels[name] for name in self._node_labels]

    def add_nonterminal(self, label: EdgeLabel):
        if label.is_terminal():
            raise Exception(f"Can't add terminal edge label {label.name()} as a nonterminal.")
            
        name = label.name()
        if name in self._nonterminals:
            if self._nonterminals[name] != label:
                raise Exception(f"There is already a nonterminal called {name}.")
        if name in self._terminals:
            raise Exception(f"Cannot have both a terminal and nonterminal with name {name}.")

        self._nonterminals[name] = label
    
    def get_nonterminal(self, name):
        return self._nonterminals[name]

    def nonterminals(self):
        return [self._nonterminals[name] for name in self._nonterminals]
    
    def add_terminal(self, label: EdgeLabel):
        if not label.is_terminal():
            raise Exception(f"Can't add nonterminal edge label {label.name()} as a terminal.")
        
        name = label.name()
        if name in self._terminals:
            if self._terminals[name] != label:
                raise Exception(f"There is already a terminal called {name}.")
        if name in self._nonterminals:
            raise Exception(f"Cannot have both a terminal and nonterminal with name {name}.")
        
        self._terminals[name] = label

    def get_terminal(self, name):
        return self._terminals[name]

    def terminals(self):
        return [self._terminals[name] for name in self._terminals]

    def get_edge_label(self, name):
        if name in self._nonterminals:
            return self._nonterminals[name]
        elif name in self._terminals:
            return self._terminals[name]
        else:
            raise KeyError(f'no such edge label {name}')
    
    def set_start_symbol(self, start: EdgeLabel):
        if start.arity() != 0:
            raise Exception("Start symbol must have arity 0.")
        self.add_nonterminal(start)
        self._start = start

    def start_symbol(self):
        return self._start
        
    # TODO: this does not check to make sure these nodes and edges haven't been used in
    #       some other rule, but maybe it should
    def add_rule(self, rule: FGGRule):
        lhs = rule.lhs()
        rhs = rule.rhs()
        
        self.add_nonterminal(lhs)
        for node in rhs.nodes():
            self.add_node_label(node.label())
        for edge in rhs.edges():
            if edge.label().is_terminal():
                self.add_terminal(edge.label())
            else:
                self.add_nonterminal(edge.label())
        
        lhs_name = lhs.name()
        if lhs_name not in self._rules:
            self._rules[lhs_name] = set()
        self._rules[lhs_name].add(rule)

    def all_rules(self):
        return [rule for nt_name in self._rules for rule in self._rules[nt_name]]
    
    def rules(self, nt_name):
        return [rule for rule in self._rules[nt_name]]
    
    def __str__(self):
        string = "Factor graph grammar with:"
        string += "\n\tNode labels:"
        for label_name in self._node_labels:
            string += f"\n\t\t{self._node_labels[label_name]}"
        string += "\n\tEdge labels:"
        for label_name in self._nonterminals:
            string += f"\n{self._nonterminals[label_name].to_string(2)}"
        for label_name in self._terminals:
            string += f"\n{self._terminals[label_name].to_string(2)}"
        string += f"\n\tStart symbol {self._start.name()}"
        string += f"\n\tProductions:"
        for nonterminal in self._rules:
            for rule in self._rules[nonterminal]:
                string += f"\n{rule.to_string(2)}"
        return string
