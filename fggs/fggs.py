__all__ = ['NodeLabel', 'EdgeLabel', 'Node', 'Edge', 'Graph', 'HRGRule', 'HRG', 'FactorGraph', 'FGG']

from typing import Optional, Iterable, Tuple, Union, Dict, Sequence, List, cast
from dataclasses import dataclass, field
from fggs.domains import Domain, FiniteDomain
from fggs.factors import Factor, FiniteFactor
import copy

@dataclass(frozen=True)
class NodeLabel:
    """A node label. This is currently just a thin wrapper around strings."""
    
    name: str
    
    def __str__(self):
        return f"NodeLabel {self.name}"


@dataclass(frozen=True, init=False)
class EdgeLabel:
    """An edge label.

    - name (str): The name of the edge label, which must be unique within an HRG.
    - node_labels (sequence of NodeLabels): If an edge has this label, its attachment nodes must have labels node_labels.
    - is_terminal (bool, optional): This label is a terminal symbol.
    - is_nonterminal (bool, optional): This label is a nonterminal symbol.
    """
    
    name: str
    node_labels: Iterable[NodeLabel]
    is_terminal: bool                #: Whether the edge label is a terminal symbol.

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
        """Whether the edge label is a nonterminal symbol."""
        return not self.is_terminal

    @property
    def arity(self):
        """The arity of the edge label (how many attachment nodes an edge with this label must have)."""
        return len(self.node_labels)

    @property
    def type(self):
        """The tuple of node labels that the attachment nodes an edge with this label must have."""
        return self.node_labels

    def __str__(self):
        return self.to_string(0)
    def to_string(self, indent=0, verbose=True):
        string = ""
        if verbose:
            string += "  "*indent
        if self.is_terminal:
            string += f"Terminal EdgeLabel {self.name}"
        else:
            string += f"Nonterminal EdgeLabel {self.name}"
        if verbose and self.arity != 0:
            string += " with type:"
            for i, node_label in enumerate(self.type):
                string += "\n  " + "  "*indent + f"{node_label}"
        return string

_id = id # hack to allow id to be used as a keyword argument below

@dataclass(frozen=True)
class Node:
    """A node of a Graph."""
    
    label: NodeLabel #: The node's label
    id: str          #: The node's id, which must be unique. If not supplied, a random one is chosen.
    persist_id: bool #: Whether the id should be saved with the Node

    def __init__(self, label: NodeLabel, id: Optional[str] = None):
        object.__setattr__(self, 'label', label)
        if id is None:
            # If no id was specified, use the object's address as a numeric id.
            # Since explicit ids are required to be strings, there can't be an
            # accidental id collision. We also set persist_id to False so that
            # if the Node is saved and loaded, it will receive a new id.
            object.__setattr__(self, 'id', _id(self))
            object.__setattr__(self, 'persist_id', False)
        else:
            if not isinstance(id, str):
                raise TypeError('explicit Node ids must be strings')
            object.__setattr__(self, 'id', id)
            object.__setattr__(self, 'persist_id', True)

    def __str__(self):
        return f"Node {self.id} with {self.label}"


@dataclass(frozen=True)
class Edge:
    """A hyperedge of a Graph."""

    label: EdgeLabel        #: The edge's label
    nodes: Tuple[Node, ...] #: The edge's attachment nodes
    id: str                 #: The edge's id, which must be unique. If not supplied, a random one is chosen.
    persist_id: bool        #: Whether the id should be saved with the Node

    def __init__(self, label: EdgeLabel, nodes: Iterable[Node], id: Optional[str] = None):
        # See Node.__init__ for further explanation of id and persist_id.
        if id is None:
            object.__setattr__(self, 'id', _id(self))
            object.__setattr__(self, 'persist_id', False)
        else:
            if not isinstance(id, str):
                raise TypeError('explicit Edge ids must be strings')
            object.__setattr__(self, 'id', id)
            object.__setattr__(self, 'persist_id', True)

        if label.type != tuple([node.label for node in nodes]):
            raise ValueError(f"Can't use edge label {label.name} with nodes labeled ({','.join(node.label.name for node in nodes)}).")

        object.__setattr__(self, 'label', label)
        object.__setattr__(self, 'nodes', tuple(nodes))
    
    def __str__(self):
        return self.to_string(0)
    def to_string(self, indent):
        string = "  "*indent
        string += f"Edge {self.id} with {self.label.to_string(verbose=False)}"
        if len(self.nodes) > 0:
            string += " connecting to:"
            for node in self.nodes:
                string += "\n  " + "  "*indent
                string += f"{node}"
        return string


class LabelingMixin:

    _node_labels: Dict[str, NodeLabel] # from names to NodeLabels
    _edge_labels: Dict[str, EdgeLabel] # from names to EdgeLabels
        
    def add_node_label(self, label: NodeLabel):
        """Adds a node label to the set of used node labels."""
        self._node_labels[label.name] = label

    def has_node_label_name(self, name: str) -> bool:
        """Returns true if there is a used node label with the given name."""
        return name in self._node_labels.keys()

    def get_node_label(self, name: str) -> NodeLabel:
        """Returns the unique used node label with the given name."""
        return self._node_labels[name]

    def node_labels(self) -> Iterable[NodeLabel]:
        """Returns a view of the node labels used."""
        return self._node_labels.values()

    def add_edge_label(self, label: EdgeLabel):
        """Adds an edge label to the set of used edge labels."""
        name = label.name
        if name in self._edge_labels.keys() and self._edge_labels[name] != label:
            raise ValueError(f"There is already an edge label called {name}.")
        self._edge_labels[name] = label
        
    def has_edge_label_name(self, name: str) -> bool:
        """Returns true if there is an edge label with the given name."""
        return name in self._edge_labels.keys()

    def get_edge_label(self, name: str) -> EdgeLabel:
        """Returns the unique used edge label with the given name."""
        return self._edge_labels[name]

    def edge_labels(self) -> Iterable[EdgeLabel]:
        """Returns a view of the edge labels used."""
        return self._edge_labels.values()
    
    def nonterminals(self) -> Iterable[EdgeLabel]:
        """Returns a copy of the list of nonterminals used."""
        return [el for el in self._edge_labels.values() if el.is_nonterminal]
    
    def terminals(self) -> Iterable[EdgeLabel]:
        """Returns a copy of the list of terminals used."""
        return [el for el in self._edge_labels.values() if el.is_terminal]


class Graph(LabelingMixin, object):
    """A hypergraph or hypergraph fragment (= hypergraph with external nodes)."""

    def __init__(self):
        self._nodes: Dict[str, Node]            = dict() # from ids to Nodes
        self._edges: Dict[str, Edge]            = dict() # from ids to Edges
        self._node_labels: Dict[str, NodeLabel] = dict() # from names to NodeLabels
        self._edge_labels: Dict[str, EdgeLabel] = dict() # from names to EdgeLabels
        self._ext: Tuple[Node, ...]             = ()
    
    def nodes(self):
        """Returns a view of the nodes in the hypergraph."""
        return self._nodes.values()
    
    def edges(self):
        """Returns a view of the hyperedges in the hypergraph."""
        return self._edges.values()

    @property
    def ext(self):
        """Tuple of external nodes."""
        return self._ext

    @ext.setter
    def ext(self, nodes: Iterable[Node]):
        """Sets the external nodes. If they are not already in the hypergraph, they are added."""
        for node in nodes:
            if node.id not in self._nodes.keys():
                self.add_node(node)
        self._ext = tuple(nodes)
    
    @property
    def arity(self):
        """Returns the number of external nodes."""
        return len(self._ext)

    @property
    def type(self):
        """Returns the tuple of node labels of the external nodes."""
        return tuple([node.label for node in self._ext])
    
    def add_node(self, node: Node):
        """Adds a node to the hypergraph."""
        if node.id in self._nodes.keys():
            raise ValueError(f"Can't have two nodes with same ID {node.id} in same Graph.")
        self.add_node_label(node.label)
        self._nodes[node.id] = node

    def has_node_id(self, nid: str):
        """Returns True iff the graph has a node with id `nid`."""
        return nid in self._nodes.keys()

    def new_node(self, name: str, id: Optional[str] = None) -> Node:
        """Convenience function for creating and adding a Node at the same time."""
        node = Node(NodeLabel(name), id=id)
        self.add_node(node)
        return node

    def remove_node(self, node: Node):
        """Removes a node from the hypergraph."""
        if node.id not in self._nodes.keys():
            raise ValueError(f'Node {node} cannot be removed because it does not belong to this Graph')
        for edge in self._edges.values():
            if node in edge.nodes:
                raise ValueError(f'Node {node} cannot be removed because it is an attachment node of Edge {edge}')
        if node in self.ext:
            raise ValueError(f'Node {node} cannot be removed because it is an external node of the Graph')
        del self._nodes[node.id]

    def add_edge(self, edge: Edge):
        """Adds a hyperedge to the hypergraph. If the attachment nodes are not already in the hypergraph, they are added."""
        if edge.id in self._edges.keys():
            raise ValueError(f"Can't have two edges with same ID {edge.id} in same Graph.")
        for node in edge.nodes:
            if node.id not in self._nodes.keys():
                self.add_node(node)
        self.add_edge_label(edge.label)
        self._edges[edge.id] = edge
        self._edge_labels[edge.label.name] = edge.label

    def has_edge_id(self, eid: str):
        """Returns True iff the graph has an edge with id `eid`."""
        return eid in self._edges.keys()

    def new_edge(self, name: str, nodes: Sequence[Node],
                 *,
                 is_terminal: bool = False, is_nonterminal: bool = False,
                 id: Optional[str] = None) -> Edge:
        """Convenience function for creating and adding an Edge at the same time."""
        edge = Edge(EdgeLabel(name, [node.label for node in nodes],
                              is_terminal=is_terminal, is_nonterminal=is_nonterminal),
                    nodes,
                    id=id)
        self.add_edge(edge)
        return edge

    def remove_edge(self, edge: Edge):
        """Removes a hyperedge from the hypergraph."""
        if edge.id not in self._edges.keys():
            raise ValueError(f'Graph does not contain Edge {edge}')
        del self._edges[edge.id]

    def copy(self):
        """Returns a copy of this Graph."""
        copy = Graph()
        copy._nodes = dict(self._nodes)
        copy._edges = dict(self._edges)
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
        string = "  "*indent + f"Graph containing:"
        if num_nodes > 0:
            for node in self._nodes.values():
                string += "\n  " + "  "*indent
                if node in self.ext:
                    string += "External "
                string += str(node)
        if num_edges > 0:
            for edge in self._edges.values():
                string += "\n" + edge.to_string(indent+1)
        return string


@dataclass
class HRGRule:
    """An HRG production.
    - lhs: The left-hand side nonterminal symbol.
    - rhs: The right-hand side hypergraph fragment.
    """

    lhs: EdgeLabel #: The left-hand side nonterminal.
    rhs: Graph     #: The right-hand side hypergraph fragment.

    def __post_init__(self):
        if self.lhs.is_terminal:
            raise Exception(f"Can't make HRG rule with terminal left-hand side.")
        if (self.lhs.type != self.rhs.type):
            raise Exception(f"Can't make HRG rule: left-hand side of type ({','.join(l.name for l in self.lhs.type)}) not compatible with right-hand side of type ({','.join(l.name for l in self.rhs.type)}).")

    def copy(self):
        """Returns a copy of this HRGRule, whose right-hand side is a copy of the original's."""
        return HRGRule(self.lhs, self.rhs.copy())

    def __str__(self):
        return self.to_string(0)
    def to_string(self, indent):
        string = "  "*indent
        string += f"HRGRule {self.lhs.name} ->\n"
        string += self.rhs.to_string(indent+1)
        return string


class HRG(LabelingMixin, object):
    """A hyperedge replacement graph grammar.
    
    Arguments:

    - start (EdgeLabel or str): Start nonterminal symbol. If start is
      a str and there isn't already an EdgeLabel by that name, it's
      assumed that its arity is zero.
    """
    
    def __init__(self, start: Union[EdgeLabel, str, None]):
        self._node_labels: Dict[str, NodeLabel] = dict()
        self._edge_labels: Dict[str, EdgeLabel] = dict()
        self._rules: Dict[EdgeLabel, List[HRGRule]] = dict()
        if start is not None:
            self.start = start # type: ignore
        else:
            self.start = None # type: ignore

    @property
    def start(self) -> EdgeLabel:
        """The start nonterminal symbol."""
        return self._start

    @start.setter
    def start(self, start: Union[EdgeLabel, str]):
        if isinstance(start, str):
            if self.has_edge_label_name(start):
                start = self.get_edge_label(start)
            else:
                start = EdgeLabel(start, [], is_nonterminal=True)
        if start.is_terminal:
            raise ValueError('Start symbol must be a nonterminal')
        self.add_edge_label(start)
        self._start = start

    def add_rule(self, rule: HRGRule):
        """Add a new production to the HRG."""
        lhs = rule.lhs
        rhs = rule.rhs
        
        self.add_edge_label(lhs)
        for node in rhs.nodes():
            self.add_node_label(node.label)
        for edge in rhs.edges():
            self.add_edge_label(edge.label)
        
        self._rules.setdefault(lhs, []).append(rule)

    def new_rule(self, lhs: str, rhs: Graph):
        """Convenience function for creating and adding a Rule at the same time."""
        lhs_el = EdgeLabel(lhs, [node.label for node in rhs.ext], is_nonterminal=True)
        rule = HRGRule(lhs_el, rhs)
        self.add_rule(rule)
        return rule

    def all_rules(self):
        """Return a copy of the list of all rules."""
        return [rule for nt_name in self._rules for rule in self._rules[nt_name]]
    
    def rules(self, lhs):
        """Return a copy of the list of all rules with left-hand side `lhs`."""
        return list(self._rules.get(lhs, []))
    
    def copy(self):
        """Returns a copy of this HRG, whose rules are all copies of the original's."""
        copy = HRG(self.start)
        copy._node_labels = self._node_labels.copy()
        copy._edge_labels = self._edge_labels.copy()
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
                self.start == other.start and
                self._node_labels == other._node_labels and
                self._edge_labels == other._edge_labels)
    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        string = "HRG with:"
        string += "\n  Node labels:"
        for label_name in self._node_labels.keys():
            string += f"\n    {self._node_labels[label_name]}"
        string += "\n  Edge labels:"
        for label_name in self._edge_labels.keys():
            string += f"\n{self._edge_labels[label_name].to_string(2)}"
        string += f"\n  Start symbol {self.start.name}"
        string += f"\n  Productions:"
        for nonterminal in self._rules:
            for rule in self._rules[nonterminal]:
                string += f"\n{rule.to_string(2)}"
        return string

    
class InterpretationMixin(LabelingMixin):
    """Methods for interpreting an HRG as an FGG or a Graph as a factor graph."""

    domains: Dict[str, Domain] # from node label names to Domains
    factors: Dict[str, Factor] # from edge label names to Factors
    
    def add_domain(self, nl: NodeLabel, dom: Domain):
        """Add mapping from NodeLabel nl to Domain dom."""
        self.add_node_label(nl)
        if nl.name in self.domains:
            raise ValueError(f"NodeLabel {nl} is already mapped")
        self.domains[nl.name] = dom

    def add_factor(self, el: EdgeLabel, fac: Factor):
        """Add mapping from EdgeLabel el to Factor fac."""
        if el.is_nonterminal:
            raise ValueError(f"Nonterminals cannot be mapped to Factors")
        self.add_edge_label(el)
        if el in self.factors:
            raise ValueError(f"EdgeLabel {el} is already mapped")
        if fac.arity != el.arity:
            raise ValueError(f'Cannot interpret EdgeLabel {el} as Factor {fac} (wrong arity)')
        for nl, dom in zip(el.node_labels, fac.domains):
            if nl.name not in self.domains:
                raise ValueError(f'Cannot interpret EdgeLabel {el} as Factor {fac} (NodeLabel {nl} not mapped)')
            elif dom != self.domains[nl.name]:
                raise ValueError(f'Cannot interpret EdgeLabel {el} as Factor {fac} (Domain {dom} != Domain {self.domains[nl.name]})')
        self.factors[el.name] = fac

    def shape(self, x: Union[Sequence[NodeLabel], Sequence[Node], EdgeLabel, Edge]):
        """Return the 'shape' of an Edge or EdgeLabel; that is, 
        the shape a tensor would need to be in order to be a
        factor for that Edge or EdgeLabel.
        """
        nls: Sequence[NodeLabel]
        if isinstance(x, Sequence):
            if len(x) > 0 and isinstance(x[0], Node):
                x = cast(Sequence[Node], x)
                nls = [node.label for node in x]
            else:
                x = cast(Sequence[NodeLabel], x)
                nls = x
        elif isinstance(x, EdgeLabel):
            nls = x.type
        else:
            nls = x.label.type
        return tuple(cast(FiniteDomain, self.domains[nl.name]).size() for nl in nls)
    
    def new_finite_domain(self, name: str, values: Sequence):
        nl = NodeLabel(name)
        dom = FiniteDomain(values)
        self.add_domain(nl, dom)
        return dom

    def new_finite_factor(self, name: str, weights):
        if not self.has_edge_label_name(name):
            raise KeyError(f"there isn't an edge label named {name}")
        el = self.get_edge_label(name)
        doms = [self.domains[nl.name] for nl in el.node_labels]
        fac = FiniteFactor(doms, weights)
        self.add_factor(el, fac)
        return fac

class FactorGraph(InterpretationMixin, Graph):
    """A factor graph."""
    
    def __init__(self):
        super().__init__()
        self.domains: Dict[str, Domain] = {}
        self.factors: Dict[str, Factor] = {}

    @staticmethod
    def from_graph(g: Graph):
        """Create a FactorGraph out of a Graph and no domains and factors."""
        fg = FactorGraph()
        for node in g.nodes():
            fg.add_node(node)
        for edge in g.edges():
            fg.add_edge(edge)
        fg._ext = tuple(g._ext)
        return fg

    def copy(self):
        """Returns a copy of this FactorGraph."""
        fg = FactorGraph()
        for node in self.nodes():
            fg.add_node(node)
        for edge in self.edges():
            fg.add_edge(edge)
        fg._ext = tuple(self._ext)
        fg.domains = copy.deepcopy(self.domains)
        fg.factors = copy.deepcopy(self.factors)
        return fg
        
class FGG(InterpretationMixin, HRG):
    """A factor graph grammar. If start is a str and there isn't already
      an EdgeLabel by that name, it's assumed that its arity is
      zero.
    """
    
    def __init__(self, start: Union[EdgeLabel, str, None]):
        super().__init__(start)
        self.domains: Dict[str, Domain] = {}
        self.factors: Dict[str, Factor] = {}

    @staticmethod
    def from_hrg(hrg: HRG):
        """Create an FGG out of an HRG and no domains and factors."""
        fgg = FGG(hrg.start)
        for r in hrg.all_rules():
            fgg.add_rule(r)
        return fgg

    def copy(self):
        """Returns a copy of this FGG."""
        fgg = FGG(self.start)
        fgg._node_labels = self._node_labels.copy()
        fgg._edge_labels = self._edge_labels.copy()
        fgg._rules = {}
        for lhs in self._rules:
            fgg._rules[lhs] = [r.copy() for r in self._rules[lhs]]
        fgg.domains = copy.deepcopy(self.domains)
        fgg.factors = copy.deepcopy(self.factors)
        return fgg
