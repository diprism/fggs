# TODO: improve the comment structure
# TODO: add static typing with mypy?
# TODO: figure out how to do type checking on functions
# TODO: change how I number nodes/edges to make pretty-printing work better
# TODO: add custom sorting functions for nodes/edges based on their IDs?



class NodeLabel:

    # name: a string
    # domain: a Domain
    def __init__(self, name, domain):
        self._name   = name
        self._domain = domain
    
    def name(self):
        return self._name
    
    def domain(self):
        return self._domain
    
    def __str__(self):
        return f"NodeLabel {self._name} with Domain {self._domain.name()}"



class EdgeLabel:
    
    # name: a string
    # is_terminal: a boolen
    # node_labels: a tuple of NodeLabels
    # factor_function: the factor function
    #       if node_labels is (nl1, nl2, ..., nlm)
    #       then this function must be 
    #           nl1.domain() X nl2.domain() X ... X nlm.domain() --> reals 
    def __init__(self, name, is_terminal, node_labels, fac=None):
        self._name        = name
        self._is_terminal = is_terminal
        self._node_labels = node_labels
        self.set_factor(fac)

    def name(self):
        return self._name
    
    def is_terminal(self):
        return self._is_terminal
        
    def arity(self):
        return len(self._node_labels)
        
    def type(self):
        return self._node_labels
    
    def factor(self):
        return self._factor
    
    def set_factor(self, fac):
        if self._is_terminal and fac is None:
            raise ValueError(f"Terminal edge label {self._name} is missing a factor.")
        if not self._is_terminal and fac is not None:
            raise ValueError(f"Nonterminal edge label {self._name} should not have a factor.")
        self._factor = fac
        
    def apply_factor(self, args):
        if not self.is_terminal():
            raise Exception(f"Cannot apply factor function because nonterminal edge label {self._name} does not have a factor function.")
        if len(args) != self.arity():
            raise Exception(f"Factor function for edge label {self._name} is not applicable to arguments {args}.")
        for i in range(self.arity()):
            if not self._node_labels[i].domain().contains(args[i]):
                raise Exception(f"Factor function for edge label {self._name} is not applicable to arguments {args}.")
        return self._factor.apply(args)

    # checks whether a label fits a list of nodes
    def is_applicable_to(self, nodes):
        if len(nodes) != self.arity():
            return False
        for i, node in enumerate(nodes):
            if node.label() != self._node_labels[i]:
                return False
        return True

    def __str__(self):
        return self.to_string(0)
    def to_string(self, indent):
        string = "\t"*indent
        if self.is_terminal():
            string += f"Terminal EdgeLabel {self._name} with arity {self.arity()}"
        else:
            string += f"Nonterminal EdgeLabel {self._name} with arity {self.arity()}"
        if self.arity() != 0:
            string += " and endpoints of type:"
            for i, node_label in enumerate(self.type()):
                string += "\n\t" + "\t"*indent + f"{i+1}. NodeLabel {node_label.name()}"
        return string

    
class Node:
    
    node_count = 0
    
    # label: a NodeLabel
    def __init__(self, label):
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
    
    # label: an EdgeLabel
    # nodes: a tuple of Nodes
    def __init__(self, label, nodes):
        Edge.edge_count += 1
        self._id     = Edge.edge_count
        if not label.is_applicable_to(nodes):
            raise Exception(f"Can't use edge label {label.name()} with this set of nodes.")
        self._label = label
        self._nodes = nodes
    
    def edge_id(self):
        return self._id
    
    def label(self):
        return self._label

    def nodes(self):
        return self._nodes
    
    def node_at(self, i):
        return self._nodes[i]
    
    def apply_factor(self):
        return self._label.apply_factor([node.value() for node in self._nodes])
    
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
    
    def add_node(self, node):
        self._nodes.add(node)
    
    def add_edge(self, edge):
        for node in edge.nodes():
            if node not in self._nodes:
                self._nodes.add(node)
        self._edges.add(edge)

    def set_ext(self, nodes):
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
    
    def __init__(self, lhs, rhs):
        if lhs.is_terminal():
            raise Exception(f"Can't make FGG rule with terminal lef-hand side.")
        if (lhs.type() != rhs.type()):
            raise Exception(f"Can't make FGG rule: left-hand side of type {lhs.type()} not compatible with right-hand side of type {rhs.type()}.")
        self._lhs = lhs
        self._rhs = rhs
    
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

    def add_node_label(self, label):
        name = label.name()
        if name in self._node_labels:
            if self._node_labels[name] != label:
                raise Exception(f"There is already a node label with name {name}.")
        self._node_labels[label.name()] = label

    def node_labels(self):
        return [self._node_labels[name] for name in self._node_labels]

    def add_nonterminal(self, label):
        if label.is_terminal():
            raise Exception(f"Can't add terminal edge label {label.name()} as a nonterminal.")
            
        name = label.name()
        if name in self._nonterminals:
            if self._nonterminals[name] != label:
                raise Exception(f"There is already a nonterminal called {name}.")
        if name in self._terminals:
            raise Exception(f"Cannot have both a terminal and nonterminal with name {name}.")

        self._nonterminals[name] = label
    
    def nonterminals(self):
        return [self._nonterminals[name] for name in self._nonterminals]
    
    def add_terminal(self, label):
        if not label.is_terminal():
            raise Exception(f"Can't add nonterminal edge label {label.name()} as a terminal.")
        
        name = label.name()
        if name in self._terminals:
            if self._terminals[name] != label:
                raise Exception(f"There is already a terminal called {name}.")
        if name in self._nonterminals:
            raise Exception(f"Cannot have both a terminal and nonterminal with name {name}.")
        
        self._terminals[name] = label

    def terminals(self):
        return [self._terminals[name] for name in self._terminals]
    
    def set_start_symbol(self, start):
        if start.arity() != 0:
            raise Exception("Start symbol must have arity 0.")
        self.add_nonterminal(start)
        self._start = start

    def start_symbol(self):
        return self._start
        
    # TODO: this does not check to make sure these nodes and edges haven't been used in
    #       some other rule, but maybe it should
    def add_rule(self, rule):
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
