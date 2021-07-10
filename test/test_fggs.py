import unittest
from dataclasses import FrozenInstanceError

from domains import Domain, FiniteDomain
from factors import Factor, ConstantFactor
from fggs import NodeLabel, EdgeLabel, Node, Edge, FactorGraph, FGGRule, FGG
import copy



class TestNodeLabel(unittest.TestCase):

    def setUp(self):
        self.dom    = FiniteDomain({1, 2, 3, 4, 5})
        self.name = "nl"
        self.nl   = NodeLabel(self.name, self.dom)

    def testImmutable(self):
        with self.assertRaises(FrozenInstanceError):
            self.nl.name = "foo"
        with self.assertRaises(FrozenInstanceError):
            self.nl.domain = FiniteDomain({"foo"})

    def testEquals(self):
        dom_eq = FiniteDomain({1, 2, 3, 4, 5})
        dom_ne = FiniteDomain({6, 7, 8, 9, 10})
        nl_eq  = NodeLabel(self.name, dom_eq)
        nl_ne  = NodeLabel(self.name, dom_ne)
        self.assertTrue(self.nl == nl_eq)
        self.assertFalse(self.nl == nl_ne)

    def testHash(self):
        dom_eq = FiniteDomain({1, 2, 3, 4, 5})
        dom_ne = FiniteDomain({6, 7, 8, 9, 10})
        nl_eq  = NodeLabel(self.name, dom_eq)
        nl_ne  = NodeLabel(self.name, dom_ne)
        d = dict()
        d[self.nl] = 5
        self.assertTrue(nl_eq in d)
        d[nl_eq] = 7
        self.assertTrue(d[self.nl] == 7)
        self.assertFalse(nl_ne in d)



class TestEdgeLabel(unittest.TestCase):
    
    def setUp(self):
        self.dom = FiniteDomain({1, 2, 3, 4, 5})
        self.nl1 = NodeLabel("nl1", self.dom)
        self.nl2 = NodeLabel("nl2", self.dom)
        self.nl3 = NodeLabel("nl3", self.dom)
        self.fac1 = ConstantFactor([self.dom]*3, 5)
        self.fac2 = ConstantFactor([self.dom]*2, 6)
        self.terminal    = EdgeLabel("t", (self.nl1, self.nl2, self.nl3), self.fac1)
        self.nonterminal = EdgeLabel("nt", (self.nl1, self.nl2))

    def test_init_bad_input(self):
        with self.assertRaises(ValueError):
            terminal_without_factor = EdgeLabel("t_missing_fac", (self.nl1, self.nl2, self.nl3), is_terminal=True)
        with self.assertRaises(ValueError):
            nonterminal_with_factor = EdgeLabel("nt_with_fac", (self.nl1, self.nl2), self.fac1, is_nonterminal=True)

        bad_dom = FiniteDomain({6, 7, 8, 9, 10})
        bad_fac = ConstantFactor([bad_dom]*2, 7)
        with self.assertRaises(ValueError):
            domain_mismatch = EdgeLabel("domain_mismatch", (self.nl1, self.nl2), bad_fac)
        with self.assertRaises(ValueError):
            arity_mismatch = EdgeLabel("arity_mismatch", (self.nl1, self.nl2), self.fac1)

    def testImmutable(self):
        with self.assertRaises(FrozenInstanceError):
            self.terminal.name = "foo"
        with self.assertRaises(FrozenInstanceError):
            self.terminal.node_labels = (self.nl1, self.nl2)
        with self.assertRaises(FrozenInstanceError):
            self.terminal.factor = self.fac2
        with self.assertRaises(FrozenInstanceError):
            self.terminal.is_terminal = False
        with self.assertRaises(FrozenInstanceError):
            self.terminal.is_nonterminal = True

    def testEquals(self):
        fac_eq = ConstantFactor([self.dom]*3, 5)
        fac_ne = ConstantFactor([self.dom]*3, 6)
        terminal_eq_fac  = EdgeLabel("t", (self.nl1, self.nl2, self.nl3), fac_eq)
        terminal_ne_name = EdgeLabel("x", (self.nl1, self.nl2, self.nl3), self.fac1)
        terminal_ne_type = EdgeLabel("t", (self.nl1, self.nl3, self.nl2), self.fac1)
        terminal_ne_fac  = EdgeLabel("t", (self.nl1, self.nl2, self.nl3), fac_ne)
        self.assertTrue(self.terminal == terminal_eq_fac)
        self.assertFalse(self.terminal == terminal_ne_name)
        self.assertFalse(self.terminal == terminal_ne_type)
        self.assertFalse(self.terminal == terminal_ne_fac)

    def testHash(self):
        fac_eq = ConstantFactor([self.dom]*3, 5)
        fac_ne = ConstantFactor([self.dom]*3, 6)
        terminal_eq_fac  = EdgeLabel("t", (self.nl1, self.nl2, self.nl3), fac_eq)
        terminal_ne_name = EdgeLabel("x", (self.nl1, self.nl2, self.nl3), self.fac1)
        terminal_ne_type = EdgeLabel("t", (self.nl1, self.nl3, self.nl2), self.fac1)
        terminal_ne_fac  = EdgeLabel("t", (self.nl1, self.nl2, self.nl3), fac_ne)
        d = dict()
        d[self.terminal] = 5
        self.assertTrue(terminal_eq_fac in d)
        self.assertFalse(terminal_ne_name in d)
        self.assertFalse(terminal_ne_type in d)
        self.assertFalse(terminal_ne_fac in d)
        d[terminal_eq_fac] = 7
        self.assertTrue(d[self.terminal] == 7)

    def test_is_terminal(self):
        self.assertTrue(self.terminal.is_terminal())
        self.assertFalse(self.nonterminal.is_terminal())
    
    def test_is_nonterminal(self):
        self.assertFalse(self.terminal.is_nonterminal())
        self.assertTrue(self.nonterminal.is_nonterminal())

    def test_arity(self):
        self.assertEqual(self.terminal.arity(), 3)
    
    def test_type(self):
        self.assertEqual(self.terminal.type(), (self.nl1, self.nl2, self.nl3))
    


class TestNode(unittest.TestCase):
    
    def setUp(self):
        self.dom = FiniteDomain({1, 2, 3, 4, 5})
        self.label = NodeLabel("label", self.dom)
        self.node1 = Node(self.label)
        self.node2 = Node(self.label, id="id2")

    def test_id(self):
        self.assertEqual(self.node2.id, "id2")




class TestEdge(unittest.TestCase):
    
    def setUp(self):
        self.dom   = FiniteDomain({1, 2, 3, 4, 5})
        self.nl1   = NodeLabel("nl1", self.dom)
        self.nl2   = NodeLabel("nl2", self.dom)
        self.node1 = Node(self.nl1)
        self.node2 = Node(self.nl2)
        
        self.fac   = ConstantFactor([self.dom]*2, 42)
        self.el1   = EdgeLabel("el1", (self.nl1, self.nl2), self.fac)
        self.el2   = EdgeLabel("el2", (self.nl2,))
        
        self.edge1 = Edge(self.el1, (self.node1, self.node2))
        self.edge2 = Edge(self.el2, (self.node2,))
    
    def test_init_bad_input(self):
        # nodes have wrong domains
        with self.assertRaises(Exception):
            bad_edge = self.Edge(self.el2, (self.node1,))
        # list of nodes has wrong arity
        with self.assertRaises(Exception):
            bad_edge = self.Edge(self.el2, (self.node2, self.node2))
    
        
class TestFactorGraph(unittest.TestCase):

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

    def test_add_node(self):
        nodes = self.graph.nodes()
        self.assertEqual(len(nodes), 2)
        self.assertTrue(self.node1 in nodes)
        self.assertTrue(self.node2 in nodes)

    def test_add_node_duplicate(self):
        # Can't add a Node already in a FactorGraph
        with self.assertRaises(ValueError):
            self.graph.add_node(self.node1)
        # nor a copy of a Node already in a FactorGraph
        new_node = Node(self.nl1, id=self.node1.id)
        with self.assertRaises(ValueError):
            self.graph.add_node(new_node)
    
    def test_add_edge(self):
        edges = self.graph.edges()
        self.assertEqual(len(edges), 2)
        self.assertTrue(self.edge1 in edges)
        self.assertTrue(self.edge2 in edges)

    def test_add_edge_duplicate(self):
        # Can't add an Edge already in a FactorGraph
        with self.assertRaises(ValueError):
            self.graph.add_edge(self.edge1)
        # nor a copy of an Edge already in a FactorGraph
        new_edge = Edge(self.el1, (self.node1, self.node2), id=self.edge1.id)
        with self.assertRaises(ValueError):
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

    def test_terminals_and_nonterminals(self):
        self.assertEqual(self.graph.terminals(), [self.edge1])
        self.assertEqual(self.graph.nonterminals(), [self.edge2])

    def test_remove_node(self):
        with self.assertRaises(ValueError):
            self.graph.remove_node(self.node1) # because nonzero degree
            
        node3 = Node(self.nl1)
        self.graph.add_node(node3)
        self.graph.remove_node(node3)
        nodes = self.graph.nodes()
        self.assertEqual(len(nodes), 2)
        self.assertTrue(self.node1 in nodes)
        self.assertTrue(self.node2 in nodes)
        self.assertTrue(node3 not in nodes)
        
        with self.assertRaises(ValueError):
            self.graph.remove_node(node3)

    def test_remove_edge(self):
        self.graph.remove_edge(self.edge1)
        edges = self.graph.edges()
        self.assertEqual(len(edges), 1)
        self.assertTrue(self.edge1 not in edges)
        self.assertTrue(self.edge2 in edges)
        with self.assertRaises(ValueError):
            self.graph.remove_edge(self.edge1)
        self.graph.add_node(self.edge1)

    def test_copy(self):
        graph = self.graph
        copy = self.graph.copy()
        self.assertNotEqual(id(graph), id(copy))
        self.assertEqual(graph, copy)

    def test_equal(self):
        self.assertEqual(self.graph, self.graph)
        
        copy = FactorGraph()
        copy.add_node(self.node1)
        copy.add_node(self.node2)
        copy.add_edge(self.edge1)
        copy.add_edge(self.edge2)
        copy.set_ext((self.node2,))

        self.assertEqual(self.graph, copy)

class TestFGGRule(unittest.TestCase):

    def setUp(self):
        dom = FiniteDomain([None])
        nl = NodeLabel("nl1", dom)
        
        node1 = Node(nl)
        node2 = Node(nl)

        fac = ConstantFactor([dom]*2, 5)
        terminal = EdgeLabel("terminal", (nl, nl), fac)
        nonterminal_mismatch = EdgeLabel("nonterminal1", (nl,))
        nonterminal_match = EdgeLabel("nonterminal2", (nl, nl))
        
        graph = FactorGraph()
        graph.add_node(node1)
        graph.add_node(node2)
        graph.set_ext((node1, node2))
        
        with self.assertRaises(Exception):
            rule = FGGRule(terminal, graph)
        with self.assertRaises(Exception):
            rule = FGGRule(nonterminal_mismatch, graph)
        self.rule = FGGRule(nonterminal_match, graph)

    def test_copy_equal(self):
        rule = self.rule
        copy = self.rule.copy()
        self.assertNotEqual(id(rule), id(copy))
        self.assertEqual(rule, copy)
        
class TestFGG(unittest.TestCase):

    def setUp(self):
        self.dom   = FiniteDomain({1, 2, 3, 4, 5})
        self.nl1   = NodeLabel("nl1", self.dom)
        self.nl2   = NodeLabel("nl2", self.dom)
        self.node1 = Node(self.nl1)
        self.node2 = Node(self.nl2)
        
        self.fac   = ConstantFactor([self.dom]*2, 42)
        self.el1   = EdgeLabel("el1", (self.nl1, self.nl2), self.fac)
        self.el2   = EdgeLabel("el2", (self.nl2,))
        self.edge1 = Edge(self.el1, (self.node1, self.node2))
        self.edge2 = Edge(self.el2, (self.node2,))
        
        self.graph = FactorGraph()
        self.graph.add_node(self.node1)
        self.graph.add_node(self.node2)
        self.graph.add_edge(self.edge1)
        self.graph.add_edge(self.edge2)
        self.graph.set_ext(tuple())
        
        self.start = EdgeLabel("S", tuple())
        self.rule = FGGRule(self.start, self.graph)
        
        self.node3 = Node(self.nl2)
        self.graph2 = FactorGraph()
        self.graph2.add_node(self.node3)
        self.graph2.set_ext((self.node3,))
        self.rule2 = FGGRule(self.el2, self.graph2)

        self.fgg = FGG()
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
        # add a node label which is a different object but
        # equivalent to an existing node label; code should
        # treat them as the same node label
        nl3 = NodeLabel("nl1", self.dom)
        self.fgg.add_node_label(nl3)
        self.assertEqual(len(self.fgg.node_labels()), 2)
    
    def test_add_node_label_bad_input(self):
        nl3 = NodeLabel("nl1", FiniteDomain([4]))
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
        nt = EdgeLabel("el2", (self.nl1, self.nl1))
        with self.assertRaises(Exception):
            self.fgg.add_nonterminal(nt)
        
        # a nonterminal with the same name as a terminal
        nt = EdgeLabel("el1", (self.nl1, self.nl1))
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
        t = EdgeLabel("el1", (self.nl1, self.nl1), self.fac)
        with self.assertRaises(Exception):
            self.fgg.add_terminal(t)
        
        # a nonterminal with the same name as a terminal
        t = EdgeLabel("el2", (self.nl1, self.nl1))
        with self.assertRaises(Exception):
            self.fgg.add_terminal(t)
        
        # it should allow you to add the same terminal a second time
        self.fgg.add_terminal(self.el1)

    def test_set_start_symbol(self):
        self.assertEqual(self.fgg.start_symbol(), self.start)

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
        new_nt = EdgeLabel("nt1", (new_nl, new_nl))
        fac = ConstantFactor([self.dom], 20)
        new_t  = EdgeLabel("nt2", (new_nl,), fac)
        
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
        
    def test_copy_equal(self):
        fgg = self.fgg
        copy = self.fgg.copy()
        self.assertNotEqual(id(fgg), id(copy))
        self.assertEqual(fgg, copy)

if __name__ == "__main__":
    unittest.main()
    
