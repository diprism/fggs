import random, string
from typing import Optional, Iterable
from domains import Domain
from factors import Factor



class NodeLabel:

    _registry = dict()

    @staticmethod
    def has_label(name):
        return name in NodeLabel._registry

    @staticmethod
    def get_label(name):
        if NodeLabel.has_label(name):
            return NodeLabel._registry[name]
        else:
            return None

    @staticmethod
    def get_all_labels():
        return list(NodeLabel._registry.keys())

    @staticmethod
    def remove_label(name):
        if name not in NodeLabel._registry:
            raise Exception(f"NodeLabel {name} cannot be deleted because it is not define.")
        del NodeLabel._registry[name]

    @staticmethod
    def clear_registry():
        for name in NodeLabel.get_all_labels():
            del NodeLabel._registry[name]

    def __init__(self, name: str, domain: Domain):
        if name in NodeLabel._registry:
            raise Exception(f"NodeLabel {name} is already defined")
        self._name   = name
        self._domain = domain
        NodeLabel._registry[name] = self
    
    def name(self):
        return self._name
    
    def domain(self):
        return self._domain
    
    def __str__(self):
        return f"NodeLabel {self._name} with Domain {self._domain}"



class EdgeLabel:

    _registry = dict()

    @staticmethod
    def has_label(name):
        return name in EdgeLabel._registry

    @staticmethod
    def get_label(name):
        if EdgeLabel.has_label(name):
            return EdgeLabel._registry[name]
        else:
            return None

    @staticmethod
    def get_all_labels():
        return list(EdgeLabel._registry.keys())

    @staticmethod
    def remove_label(name):
        if name not in EdgeLabel._registry:
            raise Exception(f"NodeLabel {name} cannot be deleted because it is not define.")
        del EdgeLabel._registry[name]

    @staticmethod
    def clear_registry():
        for name in EdgeLabel.get_all_labels():
            del EdgeLabel._registry[name]

    def __init__(self, name: str, is_terminal: bool, node_labels: Iterable[NodeLabel], fac: Optional[Factor] = None):
        if name in EdgeLabel._registry:
            raise Exception(f"EdgeLabel {name} is already defined")
        self._name        = name
        self._is_terminal = is_terminal
        self._node_labels = tuple(node_labels)
        self.set_factor(fac)
        EdgeLabel._registry[name] = self

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

    _id_registry = set()

    def __init__(self, label: NodeLabel, id: str = None):
        if id == None:
            self.set_id(self._generate_id())
        else:
            self.set_id(id)
        self._label  = label

    def _generate_id(self):
        letters = string.ascii_letters
        new_id = ''.join([random.choice(letters) for i in range(20)])
        while new_id in Node._id_registry:
            new_id = ''.join([random.choice(letters) for i in range(20)])
        return new_id
    
    def id(self):
        return self._id
    
    def set_id(self, id: str):
        Node._id_registry.add(id)
        self._id = id

    def label(self):
        return self._label
        
    def copy(self):
        """Returns a copy of this Node, including its id."""
        return Node(self._label, self._id)

    def __str__(self):
        return f"Node {self._id} with NodeLabel {self.label().name()}"



class Edge:

    _id_registry = set()

    def __init__(self, label: EdgeLabel, nodes: Iterable[Node], id: str = None):
        if id == None:
            self.set_id(self._generate_id())
        else:
            self.set_id(id)
        if label.type() != tuple([node.label() for node in nodes]):
            raise Exception(f"Can't use edge label {label.name()} with this set of nodes.")
        self._label = label
        self._nodes = tuple(nodes)

    def _generate_id(self):
        letters = string.ascii_letters
        new_id = ''.join([random.choice(letters) for i in range(20)])
        while new_id in Edge._id_registry:
            new_id = ''.join([random.choice(letters) for i in range(20)])
        return new_id

    def id(self):
        return self._id

    def set_id(self, id: str):
        Edge._id_registry.add(id)
        self._id = id
    
    def label(self):
        return self._label

    def nodes(self):
        return self._nodes
    
    def node_at(self, i):
        return self._nodes[i]

    def copy(self):
        """Returns a copy of this Edge, including its id."""
        return Edge(self._label, self._nodes, self._id)

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
                    string += f"Node {node.id()}"
        return string



class FactorGraph:

    def __init__(self):
        self._nodes    = set()
        self._node_ids = set()
        self._edges    = set()
        self._edge_ids = set()
        self._ext      = tuple()
    
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
        if node not in self._nodes and\
           node.id() in self._node_ids:
            raise Exception(f"Can't have two nodes with same ID {node.id()} in same FactorGraph.")
        self._nodes.add(node)
        self._node_ids.add(node.id())

    def add_edge(self, edge: Edge):
        if edge not in self._edges and\
           edge.id() in self._edge_ids:
            raise Exception(f"Can't have two edges with same ID {edge.id()} in same FactorGraph.")
        for node in edge.nodes():
            if node not in self._nodes:
                self._nodes.add(node)
        self._edges.add(edge)
        self._edge_ids.add(edge.id())

    def set_ext(self, nodes: Iterable[Node]):
        for node in nodes:
            if node not in self._nodes:
                self._nodes.add(node)
        self._ext = tuple(nodes)
    
    def copy(self):
        """Returns a copy of this FactorGraph, whose Nodes and Edges are also copies of the original's."""
        copy = FactorGraph()
        copy_nodes = {v.id():v.copy() for v in self._nodes}
        copy._nodes = set(copy_nodes.values())
        copy._node_ids = set(copy_nodes.keys())
        copy_edges = {}
        for e in self._edges:
            att = [copy_nodes[v.id()] for v in e.nodes()]
            copy_edges[e.id()] = Edge(e.label(), att, e.id())
        copy._edges = set(copy_edges.values())
        copy._edge_ids = set(copy_edges.keys())
        copy._ext = tuple(copy_nodes[v.id()] for v in self._ext)
        return copy

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

    def __init__(self, lhs: EdgeLabel, rhs: FactorGraph):
        if lhs.is_terminal():
            raise Exception(f"Can't make FGG rule with terminal left-hand side.")
        if (lhs.type() != rhs.type()):
            raise Exception(f"Can't make FGG rule: left-hand side of type ({','.join(l.name() for l in lhs.type())}) not compatible with right-hand side of type ({','.join(l.name() for l in rhs.type())}).")
        self._lhs = lhs
        self._rhs = rhs

    def lhs(self):
        return self._lhs
    
    def rhs(self):
        return self._rhs
    
    def copy(self):
        """Returns a copy of this FGGRule, whose right-hand side is a copy of the original's."""
        return FGGRule(self.lhs(), self.rhs().copy())

    def __str__(self):
        return self.to_string(0)
    def to_string(self, indent):
        string = "\t"*indent
        string += f"FGGRule with left-hand side {self._lhs.name()} and right-hand side as follows:\n"
        string += self._rhs.to_string(indent+1)
        return string



class FGGRepresentation:
    
    def __init__(self):
        self._node_labels  = set()     # set of NodeLabels
        self._nonterminals = set()     # set of EdgeLabels
        self._terminals    = set()     # set of EdgeLabels
        self._start        = None      # start symbol, an EdgeLabel
        self._rules        = dict()    # one set of rules for each nonterminal EdgeLabel

    def add_node_label(self, label: NodeLabel):
        self._node_labels.add(label)

    def node_labels(self):
        return list(self._node_labels)

    def add_nonterminal(self, label: EdgeLabel):
        if not label.is_nonterminal():
            raise Exception(f"Can't add terminal edge label {label.name()} as a nonterminal.")
        self._nonterminals.add(label)

    def nonterminals(self):
        return list(self._nonterminals)
    
    def add_terminal(self, label: EdgeLabel):
        if not label.is_terminal():
            raise Exception(f"Can't add nonterminal edge label {label.name()} as a terminal.")
        self._terminals.add(label)

    def terminals(self):
        return list(self._terminals)

    def set_start_symbol(self, start: EdgeLabel):
        self.add_nonterminal(start)
        self._start = start

    def start_symbol(self):
        return self._start
        
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
        
        if lhs not in self._rules:
            self._rules[lhs] = set()
        self._rules[lhs].add(rule)

    def all_rules(self):
        return [rule for nt in self._rules for rule in self._rules[nt]]
    
    def rules(self, nt):
        return [rule for rule in self._rules[nt]]
    
    def copy(self):
        """Returns a copy of this FGGRepresentation, whose rules are all copies of the original's."""
        copy = FGGRepresentation()
        copy._node_labels = self._node_labels.copy()
        copy._nonterminals = self._nonterminals.copy()
        copy._terminals = self._terminals.copy()
        copy._start = self._start
        copy._rules = {}
        for lhs in self._rules:
            copy._rules[lhs] = {r.copy() for r in self._rules[lhs]}
        return copy

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
