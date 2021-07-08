import unittest

from domains import Domain, FiniteDomain
from factors import Factor, ConstantFactor
from fgg_representation import NodeLabel, EdgeLabel, Node, Edge, FactorGraph, FGGRule, FGGRepresentation
from derivations import start_graph, replace_edges

class TestReplace(unittest.TestCase):
    def setUp(self):
        self.dom   = FiniteDomain({1, 2, 3, 4, 5})
        self.nl1   = NodeLabel("nl1", self.dom)
        self.nl2   = NodeLabel("nl2", self.dom)
        self.node1 = Node(self.nl1)
        self.node2 = Node(self.nl2)
        
        self.fac = ConstantFactor([self.dom]*2, 42)
        self.el1   = EdgeLabel("el1", (self.nl1, self.nl2), self.fac)
        self.el2   = EdgeLabel("el2", (self.nl2,))
        self.edge1 = Edge(self.el1, (self.node1, self.node2))
        self.edge2 = Edge(self.el2, (self.node2,))
        
        self.graph = FactorGraph()
        self.graph.add_node(self.node1)
        self.graph.add_node(self.node2)
        self.graph.add_edge(self.edge1)
        self.graph.add_edge(self.edge2)
        self.graph.set_ext((self.node2,))

    def test_replace(self):
        g = replace_edges(self.graph, {self.edge2:self.graph})
        self.assertEqual(sorted(v.label().name for v in g.nodes()), ['nl1', 'nl1', 'nl2'])
        self.assertEqual(sorted(e.label().name() for e in g.edges()), ['el1', 'el1', 'el2'])

class TestStartGraph(unittest.TestCase):        
    def setUp(self):
        self.start = EdgeLabel("S", tuple())

        self.fgg = FGGRepresentation()
        self.fgg.set_start_symbol(self.start)

    def test_start_graph(self):
        s = self.fgg.start_symbol()
        g = start_graph(self.fgg)
        self.assertEqual([e.label() for e in g.edges()], [s])
        self.assertEqual(len(g.nodes()), len(s.type()))
    
    
