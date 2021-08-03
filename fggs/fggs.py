__all__ = ['NodeLabel', 'EdgeLabel', 'Node', 'Edge', 'Graph', 'HRGRule', 'HRG', 'Interpretation', 'FactorGraph', 'FGG']

import random, string
from typing import Optional, Iterable, Tuple
from dataclasses import dataclass, field
from fggs.domains import Domain
from fggs.factors import Factor


@dataclass(frozen=True)
class NodeLabel:
    name: str
    
    def __str__(self):
        return f"NodeLabel {self.name}"


@dataclass(frozen=True, init=False)
class EdgeLabel:
    """An edge label.

    name (str): The name of the edge label, which must be unique within an HRG.
    node_labels (sequence of NodeLabels): If an edge has this label, its attachment nodes must have labels node_labels.
    is_terminal (bool, optional): This label is a terminal symbol.
    is_nonterminal (bool, optional): This label is a nonterminal symbol.
    """
    
    name: str
    node_labels: Iterable[NodeLabel]
    is_terminal: bool

    def __init__(self, name: str,
                 node_labels: Iterable[NodeLabel],
                 *,
                 is_nonterminal: bool = False,
                 is_terminal: bool = False):
        object.__setattr__(self, 'name', name)
        if not isinstance(node_labels, tuple):
            node_labels = tuple(node_labels)
        object.__setattr__(self, 'node_labels', node_labels)
        if is_terminal and is_nonterminal:
            raise ValueError("An EdgeLabel can't be both terminal and nonterminal")
        if not is_terminal and not is_nonterminal:
            raise ValueError("An EdgeLabel must be either terminal or nonterminal")
        object.__setattr__(self, 'is_terminal', is_terminal)

    @property
    def is_nonterminal(self):
        return not self.is_terminal

    def arity(self):
        return len(self.node_labels)
        
    def type(self):
        return self.node_labels

    def __str__(self):
        return self.to_string(0)
    def to_string(self, indent):
        string = "\t"*indent
        if self.is_terminal:
            string += f"Terminal EdgeLabel {self.name} with arity {self.arity()}"
        else:
            string += f"Nonterminal EdgeLabel {self.name} with arity {self.arity()}"
        if self.arity() != 0:
            string += " and endpoints of type:"
            for i, node_label in enumerate(self.type()):
                string += "\n\t" + "\t"*indent + f"{i+1}. NodeLabel {node_label.name}"
        return string


def _generate_id():
    letters = string.ascii_letters
    new_id = ''.join([random.choice(letters) for i in range(20)])
    return new_id


@dataclass(frozen=True)
class Node:
    
    label: NodeLabel
    id: str = None

    def __post_init__(self):
        if self.id == None:
            object.__setattr__(self, 'id', _generate_id())

    def __str__(self):
        return f"Node {self.id} with NodeLabel {self.label}"


@dataclass(frozen=True)
class Edge:

    label: EdgeLabel
    nodes: Iterable[NodeLabel]
    id: str = None

    def __post_init__(self):
        if self.id == None:
            object.__setattr__(self, 'id', _generate_id())

        if self.label.type() != tuple([node.label for node in self.nodes]):
            raise ValueError(f"Can't use edge label {self.label.name} with this set of nodes.")
        if not isinstance(self.nodes, tuple):
            object.__setattr__(self, 'nodes', tuple(self.nodes))
    
    def __str__(self):
        return self.to_string(0, True)
    def to_string(self, indent, verbose):
        arity = len(self.nodes)
        string = "\t"*indent
        string += f"Edge {self.id} with EdgeLabel {self.label}, connecting to {arity} nodes"
        if arity > 0:
            string += ":"
            for node in self.nodes:
                string += "\n\t" + "\t"*indent
                if verbose:
                    string += f"{node}"
                else:
                    string += f"Node {node.id}"
        return string


class Graph:

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
    
    def terminals(self):
        return [edge for edge in self._edges if edge.label.is_terminal]

    def nonterminals(self):
        return [edge for edge in self._edges if edge.label.is_nonterminal]

    def ext(self):
        return self._ext
    
    def arity(self):
        return len(self._ext)
    
    def type(self):
        return tuple([node.label for node in self._ext])
    
    def add_node(self, node: Node):
        if node.id in self._node_ids:
            raise ValueError(f"Can't have two nodes with same ID {node.id} in same Graph.")
        self._nodes.add(node)
        self._node_ids.add(node.id)

    def remove_node(self, node: Node):
        if node not in self._nodes:
            raise ValueError(f'Node {node} cannot be removed because it does not belong to this Graph')
        for edge in self._edges:
            if node in edge.nodes:
                raise ValueError(f'Node {node} cannot be removed because it is an attachment node of Edge {edge}')
        self._nodes.remove(node)
        self._node_ids.remove(node.id)

    def add_edge(self, edge: Edge):
        if edge.id in self._edge_ids:
            raise ValueError(f"Can't have two edges with same ID {edge.id} in same Graph.")
        for node in edge.nodes:
            if node not in self._nodes:
                self._nodes.add(node)
        self._edges.add(edge)
        self._edge_ids.add(edge.id)

    def remove_edge(self, edge: Edge):
        if edge not in self._edges:
            raise ValueError(f'Graph does not contain Edge {edge}')
        self._edges.remove(edge)
        self._edge_ids.remove(edge.id)

    def set_ext(self, nodes: Iterable[Node]):
        for node in nodes:
            if node not in self._nodes:
                self._nodes.add(node)
        self._ext = tuple(nodes)
    
    def copy(self):
        """Returns a copy of this Graph."""
        copy = Graph()
        copy._nodes = set(self._nodes)
        copy._node_ids = set(self._node_ids)
        copy._edges = set(self._edges)
        copy._edge_ids = set(self._edge_ids)
        copy._ext = tuple(self._ext)
        return copy

    def __eq__(self, other):
        """Tests if two Graphs are equal, including their Node and Edge ids.

        Runs in O(|V|+|E|) time because the ids are required to be equal."""
        return (isinstance(other, Graph) and
                self._nodes == other._nodes and
                self._edges == other._edges and
                self._ext == other._ext)
    def __ne__(self, other):
        return not self.__eq__(other)

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


class HRGRule:

    def __init__(self, lhs: EdgeLabel, rhs: Graph):
        if lhs.is_terminal:
            raise Exception(f"Can't make HRG rule with terminal left-hand side.")
        if (lhs.type() != rhs.type()):
            raise Exception(f"Can't make HRG rule: left-hand side of type ({','.join(l.name for l in lhs.type())}) not compatible with right-hand side of type ({','.join(l.name for l in rhs.type())}).")
        self._lhs = lhs
        self._rhs = rhs

    def lhs(self):
        return self._lhs
    
    def rhs(self):
        return self._rhs
    
    def copy(self):
        """Returns a copy of this HRGRule, whose right-hand side is a copy of the original's."""
        return HRGRule(self.lhs(), self.rhs().copy())

    def __eq__(self, other):
        return (isinstance(other, HRGRule) and
                self._lhs == other._lhs and
                self._rhs == other._rhs)
    def __ne__(self, other):
        return not self.__eq__(self, other)

    def __str__(self):
        return self.to_string(0)
    def to_string(self, indent):
        string = "\t"*indent
        string += f"HRGRule with left-hand side {self._lhs.name} and right-hand side as follows:\n"
        string += self._rhs.to_string(indent+1)
        return string


class HRG:
    
    def __init__(self):
        self._node_labels  = dict()    # map from names to NodeLabels
        self._nonterminals = dict()    # map from names to EdgeLabels
        self._terminals    = dict()    # map from names to EdgeLabels
        self._start        = None      # start symbol, an EdgeLabel which has arity 0
        self._rules        = dict()    # one list of rules for each nonterminal edge label

    def add_node_label(self, label: NodeLabel):
        self._node_labels[label.name] = label

    def has_node_label(self, name):
        return name in self._node_labels

    def get_node_label(self, name):
        return self._node_labels[name]

    def node_labels(self):
        return [self._node_labels[name] for name in self._node_labels]

    def add_nonterminal(self, label: EdgeLabel):
        if label.is_terminal:
            raise Exception(f"Can't add terminal edge label {label.name} as a nonterminal.")
            
        name = label.name
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
        if not label.is_terminal:
            raise Exception(f"Can't add nonterminal edge label {label.name} as a terminal.")
        
        name = label.name
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

    def has_edge_label(self, name):
        return name in self._nonterminals or\
               name in self._terminals

    def get_edge_label(self, name):
        if name in self._nonterminals:
            return self._nonterminals[name]
        elif name in self._terminals:
            return self._terminals[name]
        else:
            raise KeyError(f'no such edge label {name}')

    def edge_labels(self):
        return self.nonterminals() + self.terminals()

    def set_start_symbol(self, start: EdgeLabel):
        self.add_nonterminal(start)
        self._start = start

    def start_symbol(self):
        return self._start

    def add_rule(self, rule: HRGRule):
        lhs = rule.lhs()
        rhs = rule.rhs()
        
        self.add_nonterminal(lhs)
        for node in rhs.nodes():
            self.add_node_label(node.label)
        for edge in rhs.edges():
            if edge.label.is_terminal:
                self.add_terminal(edge.label)
            else:
                self.add_nonterminal(edge.label)
        
        self._rules.setdefault(lhs, []).append(rule)

    def all_rules(self):
        return [rule for nt_name in self._rules for rule in self._rules[nt_name]]
    
    def rules(self, lhs):
        return list(self._rules[lhs])
    
    def copy(self):
        """Returns a copy of this HRG, whose rules are all copies of the original's."""
        copy = HRG()
        copy._node_labels = self._node_labels.copy()
        copy._nonterminals = self._nonterminals.copy()
        copy._terminals = self._terminals.copy()
        copy._start = self._start
        copy._rules = {}
        for lhs in self._rules:
            copy._rules[lhs] = [r.copy() for r in self._rules[lhs]]
        return copy

    def __eq__(self, other):
        """Return True iff self and other are equal. If X is a nonterminal and
        self.rules(X) and other.rules(X) have the same rules but in a
        different order, then self and other are *not* considered
        equal."""
        return (isinstance(other, HRG) and
                self._rules == other._rules and
                self._start == other._start and
                self._node_labels == other._node_labels and
                self._nonterminals == other._nonterminals and
                self._terminals == other._terminals)
    def __ne__(self, other):
        return not self.__eq__(other)

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
        string += f"\n\tStart symbol {self._start.name}"
        string += f"\n\tProductions:"
        for nonterminal in self._rules:
            for rule in self._rules[nonterminal]:
                string += f"\n{rule.to_string(2)}"
        return string

    
class Interpretation:
    def __init__(self):
        self.domains = {}
        self.factors = {}

    def can_interpret(self, g: HRG):
        """Test whether this Interpretation is compatible with HRG g."""
        return (set(self.domains.keys()).issuperset(set(g.node_labels())) and
                set(self.factors.keys()).issuperset(set(g.terminals())))
    
    def add_domain(self, nl: NodeLabel, dom: Domain):
        """Add mapping from NodeLabel nl to Domain dom."""
        if nl in self.domains:
            raise ValueError(f"NodeLabel {nl} is already mapped")
        self.domains[nl] = dom

    def add_factor(self, el: EdgeLabel, fac: Factor):
        if el.is_nonterminal:
            raise ValueError(f"Nonterminals cannot be mapped to Factors")
        if el in self.factors:
            raise ValueError(f"EdgeLabel {el} is already mapped")
        doms = list(fac.domains())
        if len(doms) != len(el.node_labels):
            raise ValueError(f'Cannot interpret EdgeLabel {el} as Factor {fac} (wrong arity)')
        for nl, dom in zip(el.node_labels, doms):
            if nl not in self.domains:
                raise ValueError(f'Cannot interpret EdgeLabel {el} as Factor {fac} (NodeLabel {dom} not mapped)')
            elif dom != self.domains[nl]:
                raise ValueError(f'Cannot interpret EdgeLabel {el} as Factor {fac} (Domain {dom} != Domain {self.domains[nl]})')
        self.factors[el] = fac
        
    
class FactorGraph:    
    def __init__(self, graph: Graph, interp: Interpretation):
        self.graph = graph
        self.interp = interp

        
class FGG:
    def __init__(self, grammar: HRG, interp: Interpretation):
        self.grammar = grammar
        self.interp = interp
        
