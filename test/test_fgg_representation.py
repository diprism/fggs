import unittest

from domains import Domain, FiniteDomain
from factors import Factor, ConstantFactor
from fgg_representation import NodeLabel, EdgeLabel, Node, Edge, FactorGraph, FGGRule, FGGRepresentation



class TestEdgeLabel(unittest.TestCase):
    
    def setUp(self):
        self.dom = FiniteDomain({1, 2, 3, 4, 5})
        self.nl1 = NodeLabel("nl1", self.dom)
        self.nl2 = NodeLabel("nl2", self.dom)
        self.nl3 = NodeLabel("nl3", self.dom)
        self.fac1 = ConstantFactor([self.dom]*3, 5)
        self.fac2 = ConstantFactor([self.dom]*2, 6)
        self.terminal    = EdgeLabel("t", True, (self.nl1, self.nl2, self.nl3), self.fac1)
        self.nonterminal = EdgeLabel("nt", False, (self.nl1, self.nl2))

    def test_init_bad_input(self):
        with self.assertRaises(ValueError):
            terminal_without_factor = EdgeLabel("t_missing_fac", True, (self.nl1, self.nl2, self.nl3), None)
        with self.assertRaises(ValueError):
            nonterminal_with_factor = EdgeLabel("nt_with_fac", False, (self.nl1, self.nl2), self.fac1)

        bad_dom = FiniteDomain({6, 7, 8, 9, 10})
        bad_fac = ConstantFactor([bad_dom]*2, 7)
        with self.assertRaises(ValueError):
            domain_mismatch = EdgeLabel("domain_mismatch", True, (self.nl1, self.nl2), bad_fac)
        with self.assertRaises(ValueError):
            arity_mismatch = EdgeLabel("arity_mismatch", True, (self.nl1, self.nl2), self.fac1)

    def test_is_terminal(self):
        self.assertTrue(self.terminal.is_terminal())
        self.assertFalse(self.nonterminal.is_terminal())
    
    def test_arity(self):
        self.assertEqual(self.terminal.arity(), 3)
    
    def test_type(self):
        self.assertEqual(self.terminal.type(), (self.nl1, self.nl2, self.nl3))
    
    def test_set_factor(self):
        self.assertEqual(self.terminal.factor(), self.fac1)
        new_fac = ConstantFactor([self.dom]*3, 8)
        self.terminal.set_factor(new_fac)
        self.assertEqual(self.terminal.factor(), new_fac)

    def test_set_factor_bad_input(self):
        with self.assertRaises(ValueError):
            self.terminal.set_factor(None)
        with self.assertRaises(ValueError):
            self.nonterminal.set_factor(self.fac1)

        bad_dom = FiniteDomain({6, 7, 8, 9, 10})
        bad_fac = ConstantFactor([bad_dom]*3, 147)
        with self.assertRaises(ValueError):
            self.terminal.set_factor(bad_fac)
        with self.assertRaises(ValueError):
            self.terminal.set_factor(self.fac2)



class TestNode(unittest.TestCase):
    
    def setUp(self):
        self.dom = FiniteDomain({1, 2, 3, 4, 5})
        self.label = NodeLabel("label", self.dom)
        self.node1 = Node(self.label)
        self.node2 = Node(self.label)

    def test_value(self):
        self.assertFalse(self.node1.has_value())
        
        self.node1.set_value(4)
        self.assertTrue(self.node1.has_value())
        self.node1.unset_value()
        self.assertFalse(self.node1.has_value())
        
        self.node1.set_value(4)
        self.assertTrue(self.node1.has_value())
        self.node1.set_value(None)
        self.assertFalse(self.node1.has_value())
    
    def test_set_value_bad_input(self):
        with self.assertRaises(Exception):
            self.node1.set_value(6)



class TestEdge(unittest.TestCase):
    
    def setUp(self):
        self.dom   = FiniteDomain({1, 2, 3, 4, 5})
        self.nl1   = NodeLabel("nl1", self.dom)
        self.nl2   = NodeLabel("nl2", self.dom)
        self.node1 = Node(self.nl1)
        self.node2 = Node(self.nl2)
        
        self.fac   = ConstantFactor([self.dom]*2, 42)
        self.el1   = EdgeLabel("el1", True, (self.nl1, self.nl2), self.fac)
        self.el2   = EdgeLabel("el2", False, (self.nl2,))
        
        self.edge1 = Edge(self.el1, (self.node1, self.node2))
        self.edge2 = Edge(self.el2, (self.node2,))
    
    def test_init_bad_input(self):
        # nodes have wrong domains
        with self.assertRaises(Exception):
            bad_edge = self.Edge(self.el2, (self.node1,))
        # list of nodes has wrong arity
        with self.assertRaises(Exception):
            bad_edge = self.Edge(self.el2, (self.node2, self.node2))
    
    def test_node_at(self):
        self.assertEqual(self.edge1.node_at(1), self.node2)
    
    def test_apply_factor(self):
        # edge is a nonterminal
        with self.assertRaises(Exception):
            self.edge2.apply_factor()
        # node values not set
        with self.assertRaises(Exception):
            self.edge1.apply_factor()
        # node values set
        self.node1.set_value(1)
        self.node2.set_value(2)
        self.assertEqual(self.edge1.apply_factor(), 42)



class TestFactorGraph(unittest.TestCase):

    def setUp(self):
        self.dom   = FiniteDomain({1, 2, 3, 4, 5})
        self.nl1   = NodeLabel("nl1", self.dom)
        self.nl2   = NodeLabel("nl2", self.dom)
        self.node1 = Node(self.nl1)
        self.node2 = Node(self.nl2)
        
        self.fac = ConstantFactor([self.dom]*2, 42)
        self.el1   = EdgeLabel("el1", True, (self.nl1, self.nl2), self.fac)
        self.el2   = EdgeLabel("el2", False, (self.nl2,))
        self.edge1 = Edge(self.el1, (self.node1, self.node2))
        self.edge2 = Edge(self.el2, (self.node2,))
        
        self.graph = FactorGraph()
        self.graph.add_node(self.node1)
        self.graph.add_node(self.node2)
        self.graph.add_edge(self.edge1)
        self.graph.add_edge(self.edge2)
        self.graph.set_ext((self.node2,))

    def test_add_node(self):
        nodes = self.graph.nodes()
        self.assertEqual(len(nodes), 2)
        self.assertTrue(self.node1 in nodes)
        self.assertTrue(self.node2 in nodes)

    def test_add_node_duplicate(self):
        # it's fine to add the same node twice
        self.graph.add_node(self.node1)
        self.assertEqual(len(self.graph.nodes()), 2)
        # can't add two different nodes with the same id though
        id  = self.node1.id()
        new_node = Node(self.nl1, id=id)
        with self.assertRaises(Exception):
            self.graph.add_node(new_node)
    
    def test_add_edge(self):
        edges = self.graph.edges()
        self.assertEqual(len(edges), 2)
        self.assertTrue(self.edge1 in edges)
        self.assertTrue(self.edge2 in edges)

    def test_add_edge_duplicate(self):
        # it's fine to add the same edge twice
        self.graph.add_edge(self.edge1)
        self.assertEqual(len(self.graph.edges()), 2)
        # can't add two different edges with the same id though
        id  = self.edge1.id()
        new_edge = Edge(self.el1, (self.node1, self.node2), id=id)
        with self.assertRaises(Exception):
            self.graph.add_edge(new_edge)
    
    def test_set_ext(self):
        ext = self.graph.ext()
        self.assertEqual(ext, (self.node2,))
    
    def test_add_nodes_implicitly(self):
        node3 = Node(self.nl1)
        edge3 = Edge(self.el1, (node3, self.node2))
        self.graph.add_edge(edge3)
        self.assertTrue(node3 in self.graph.nodes())
        
        node4 = Node(self.nl1)
        self.graph.set_ext((self.node2, node4))
        self.assertTrue(node4 in self.graph.nodes())

    def test_arity_and_type(self):
        self.assertEqual(self.graph.arity(), 1)
        self.assertEqual(self.graph.type(), (self.nl2,))



class TestFGGRule(unittest.TestCase):

    def test_init(self):
        dom = FiniteDomain([None])
        nl = NodeLabel("nl1", dom)
        
        node1 = Node(nl)
        node2 = Node(nl)

        fac = ConstantFactor([dom]*2, 5)
        terminal = EdgeLabel("terminal", True, (nl, nl), fac)
        nonterminal_mismatch = EdgeLabel("nonterminal1", False, (nl,))
        nonterminal_match = EdgeLabel("nonterminal2", False, (nl, nl))
        
        graph = FactorGraph()
        graph.add_node(node1)
        graph.add_node(node2)
        graph.set_ext((node1, node2))
        
        with self.assertRaises(Exception):
            rule = FGGRule(terminal, graph)
        with self.assertRaises(Exception):
            rule = FGGRule(nonterminal_mismatch, graph)
        rule = FGGRule(nonterminal_match, graph)



class TestFGGRepresentation(unittest.TestCase):

    def setUp(self):
        self.dom   = FiniteDomain({1, 2, 3, 4, 5})
        self.nl1   = NodeLabel("nl1", self.dom)
        self.nl2   = NodeLabel("nl2", self.dom)
        self.node1 = Node(self.nl1)
        self.node2 = Node(self.nl2)
        
        self.fac   = ConstantFactor([self.dom]*2, 42)
        self.el1   = EdgeLabel("el1", True, (self.nl1, self.nl2), self.fac)
        self.el2   = EdgeLabel("el2", False, (self.nl2,))
        self.edge1 = Edge(self.el1, (self.node1, self.node2))
        self.edge2 = Edge(self.el2, (self.node2,))
        
        self.graph = FactorGraph()
        self.graph.add_node(self.node1)
        self.graph.add_node(self.node2)
        self.graph.add_edge(self.edge1)
        self.graph.add_edge(self.edge2)
        self.graph.set_ext(tuple())
        
        self.start = EdgeLabel("S", False, tuple())
        self.rule = FGGRule(self.start, self.graph)
        
        self.node3 = Node(self.nl2)
        self.graph2 = FactorGraph()
        self.graph2.add_node(self.node3)
        self.graph2.set_ext((self.node3,))
        self.rule2 = FGGRule(self.el2, self.graph2)

        self.fgg = FGGRepresentation()
        self.fgg.add_node_label(self.nl1)
        self.fgg.add_node_label(self.nl2)
        self.fgg.add_terminal(self.el1)
        self.fgg.add_nonterminal(self.el2)
        self.fgg.add_nonterminal(self.start)
        self.fgg.set_start_symbol(self.start)
        self.fgg.add_rule(self.rule)
        self.fgg.add_rule(self.rule2)

    def test_add_node_label(self):
        node_labels = self.fgg.node_labels()
        self.assertEqual(len(node_labels), 2)
        self.assertTrue(self.nl1 in node_labels)
        self.assertTrue(self.nl2 in node_labels)
    
    def test_add_node_label_bad_input(self):
        nl3 = NodeLabel("nl1", self.dom)
        with self.assertRaises(Exception):
            self.fgg.add_node_label(nl3)

    def test_add_nonterminal(self):
        nonterminals = self.fgg.nonterminals()
        self.assertEqual(len(nonterminals), 2)
        self.assertTrue(self.start in nonterminals)
        self.assertTrue(self.el2 in nonterminals)
    
    def test_add_nonterminal_bad_input(self):
        # try adding a terminal
        with self.assertRaises(Exception):
            self.fgg.add_nonterminal(self.el1)
        
        # reusing a nonterminal name
        nt = EdgeLabel("el2", False, (self.nl1, self.nl1))
        with self.assertRaises(Exception):
            self.fgg.add_nonterminal(nt)
        
        # a nonterminal with the same name as a terminal
        nt = EdgeLabel("el1", False, (self.nl1, self.nl1))
        with self.assertRaises(Exception):
            self.fgg.add_nonterminal(nt)
        
        # it should allow you to add the same nt a second time
        self.fgg.add_nonterminal(self.start)

    def test_add_terminal(self):
        terminals = self.fgg.terminals()
        self.assertEqual(len(terminals), 1)
        self.assertTrue(self.el1 in terminals)
        
    def test_add_terminal_bad_input(self):
        # try adding a nonterminal
        with self.assertRaises(Exception):
            self.fgg.add_terminal(self.el2)
        
        # reusing a terminal name
        t = EdgeLabel("el1", True, (self.nl1, self.nl1), self.fac)
        with self.assertRaises(Exception):
            self.fgg.add_terminal(t)
        
        # a nonterminal with the same name as a terminal
        t = EdgeLabel("el2", True, (self.nl1, self.nl1), self.fac)
        with self.assertRaises(Exception):
            self.fgg.add_terminal(t)
        
        # it should allow you to add the same terminal a second time
        self.fgg.add_terminal(self.el1)

    def test_set_start_symbol(self):
        self.assertEqual(self.fgg.start_symbol(), self.start)
    
    def test_set_start_symbol_bad_input(self):
        with self.assertRaises(Exception):
            self.fgg.set_start_symbol(self.el2)
    
    def test_add_rule(self):
        all_rules = self.fgg.all_rules()
        self.assertEqual(len(all_rules), 2)
        self.assertTrue(self.rule in all_rules)
        self.assertTrue(self.rule2 in all_rules)
        
        start_rules = self.fgg.rules("S")
        self.assertEqual(len(start_rules), 1)
        self.assertTrue(self.rule in start_rules)

    def test_implicitly_add_node_and_edge_labels(self):
        new_nl = NodeLabel("nl3", self.dom)
        new_nt = EdgeLabel("nt1", False, (new_nl, new_nl))
        fac = ConstantFactor([self.dom], 20)
        new_t  = EdgeLabel("nt2", True, (new_nl,), fac)
        
        new_node1 = Node(new_nl)
        new_node2 = Node(new_nl)
        new_edge  = Edge(new_t, (new_node1,))
        new_graph = FactorGraph()
        new_graph.add_node(new_node1)
        new_graph.add_node(new_node2)
        new_graph.add_edge(new_edge)
        new_graph.set_ext((new_node1, new_node2))
        new_rule  = FGGRule(new_nt, new_graph)
        
        self.fgg.add_rule(new_rule)
        
        self.assertTrue(new_nl in self.fgg.node_labels())
        self.assertTrue(new_nt in self.fgg.nonterminals())
        self.assertTrue(new_t  in self.fgg.terminals())
        
if __name__ == "__main__":
    unittest.main()
    
