# TODO: write unit tests
# TODO: improve the comment structure
# TODO: add static typing with mypy?
# TODO: add __str__ methods for everything
# TODO: improve how factor functions are defined so they can be updated later / parameters be changed



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



class EdgeLabel:
    
    # name: a string
    # is_terminal: a boolen
    # node_labels: a tuple of NodeLabels
    # factor_function: the factor function
    #       if node_labels is (nl1, nl2, ..., nlm)
    #       then this function must be 
    #           nl1.domain() X nl2.domain() X ... X nlm.domain() --> reals 
    def __init__(self, name, is_terminal, node_labels, factor_function):
        self._name        = name
        self._is_terminal = is_terminal
        self._arity       = len(node_labels)
        self._node_labels = node_labels
        if is_terminal and factor_function == None:
            raise Exception(f"Terminal edge label {name} is missing a factor function.")
        if not is_terminal and factor_function != None:
            raise Exception(f"Nonterminal edge label {name} should not have a factor function.")
        self._factor_func = factor_function

    def name(self):
        return self._name
    
    def is_terminal(self):
        return self._is_terminal
        
    def arity(self):
        return self._arity
        
    def apply_factor_function(self, args):
        if len(args) != self._arity:
            return None
        for i in range(self._arity):
            if not self._node_labels[i].domain().contains(args[i]):
                return None
        return self._factor_func(*args) # the * unpacks the list

    # checks whether a label fits a set of nodes
    def is_applicable_to(self, nodes):
        if len(nodes) != self._arity:
            return False
        for i, node in enumerate(nodes):
            if node.label() != self._node_labels[i]:
                return False
        return True



class Node:
    
    # label: a NodeLabel
    def __init__(self, label):
        self._label  = label
        self._value  = None
        
    def label(self):
        return self._label
    
    def has_value(self):
        return self._value != None
    
    def value(self):
        return self._value
    
    def set_value(self, val):
        if not self._label.domain().contains(val):
            raise Exception(f"Node with label {self._label.name()} cannot take value {val}.")
        self._value = val
    
    def unset_value(self):
        self._value = None



class Edge:
    
    # label: an EdgeLabel
    # nodes: a tuple of Nodes
    def __init__(self, label, nodes):
        if not label.is_applicable_to(nodes):
            raise Exception(f"Can't use edge label {label.name()} in this context.")
        self._label = label
        self._nodes = nodes
        self._arity = len(nodes)
    
    def label(self):
        return self._label

    def arity(self):
        return self._arity
    
    def nodes(self):
        return self._nodes
    
    def node_at(self, i):
        return self._nodes[i]
    
    def apply_factor_function(self):
        self._label.apply_factor_function([node.value() for node in self._nodes])



class FactorGraph:

    def __init__(self):
        self._nodes = set()
        self._edges = set()
        self._ext   = set()
    
    def nodes(self):
        return list(self._nodes)
    
    def edges(self):
        return list(self._edges)
    
    def ext(self):
        return list(self._ext)
    
    def add_node(self, node):
        self._nodes.add(node)
    
    def add_edge(self, edge):
        for node in edge.nodes():
            if node not in self._nodes:
                self._nodes.add(node)
        self._edges.add(edge)

    def add_ext_node(self, node):
        self._nodes.add(node)
        self._ext.add(node)
    
    

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
    
    def add_nonterminal(self, label):
        name = label.name()
        if name in self._nonterminals:
            if self._nonterminals[name] != label:
                raise Exception(f"There is already a nonterminal called {name}.")
        if name in self._terminals:
            raise Exception(f"Cannot have both a terminal and nonterminal with name {name}.")
        self._nonterminals[name] = label
    
    def add_terminal(self, label):
        name = label.name()
        if name in self._terminals:
            if self._terminals[name] != label:
                raise Exception(f"There is already a terminal called {name}.")
        if name in self._nonterminals:
            raise Exception(f"Cannot have both a terminal and nonterminal with name {name}.")
        self._terminals[name] = label

    def set_start_symbol(self, start):
        if start.arity() != 0:
            raise Exception("Start symbol must have arity 0.")
        self.add_nonterminal(start)
        self._start = start
    
    def add_rule(self, lhs, rhs):
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
        self._rules[lhs].add(rhs)


