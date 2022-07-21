import unittest
from dataclasses import FrozenInstanceError

from fggs import *
import copy



class TestNodeLabel(unittest.TestCase):

    def setUp(self):
        self.name = "nl"
        self.nl   = NodeLabel(self.name)

    def testImmutable(self):
        with self.assertRaises(FrozenInstanceError):
            self.nl.name = "foo"

    def testEquals(self):
        nl_eq  = NodeLabel(self.name)
        nl_ne  = NodeLabel(self.name + "*")
        self.assertTrue(self.nl == nl_eq)
        self.assertFalse(self.nl == nl_ne)

    def testHash(self):
        nl_eq  = NodeLabel(self.name)
        nl_ne  = NodeLabel(self.name + "*")
        d = dict()
        d[self.nl] = 5
        self.assertTrue(nl_eq in d)
        d[nl_eq] = 7
        self.assertTrue(d[self.nl] == 7)
        self.assertFalse(nl_ne in d)



class TestEdgeLabel(unittest.TestCase):
    
    def setUp(self):
        self.nl1 = NodeLabel("nl1")
        self.nl2 = NodeLabel("nl2")
        self.nl3 = NodeLabel("nl3")
        self.terminal    = EdgeLabel("t", (self.nl1, self.nl2, self.nl3), is_terminal=True)
        self.nonterminal = EdgeLabel("nt", (self.nl1, self.nl2), is_nonterminal=True)

    def testImmutable(self):
        with self.assertRaises(FrozenInstanceError):
            self.terminal.name = "foo"
        with self.assertRaises(FrozenInstanceError):
            self.terminal.node_labels = (self.nl1, self.nl2)
        with self.assertRaises(FrozenInstanceError):
            self.terminal.is_terminal = False
        with self.assertRaises(FrozenInstanceError):
            self.terminal.is_nonterminal = True

    def testEquals(self):
        terminal_eq = EdgeLabel("t", (self.nl1, self.nl2, self.nl3), is_terminal=True)
        terminal_ne_name = EdgeLabel("x", (self.nl1, self.nl2, self.nl3), is_terminal=True)
        terminal_ne_type = EdgeLabel("t", (self.nl1, self.nl3, self.nl2), is_terminal=True)
        self.assertTrue(self.terminal == terminal_eq)
        self.assertFalse(self.terminal == terminal_ne_name)
        self.assertFalse(self.terminal == terminal_ne_type)

    def testHash(self):
        terminal_eq = EdgeLabel("t", (self.nl1, self.nl2, self.nl3), is_terminal=True)
        terminal_ne_name = EdgeLabel("x", (self.nl1, self.nl2, self.nl3), is_terminal=True)
        terminal_ne_type = EdgeLabel("t", (self.nl1, self.nl3, self.nl2), is_terminal=True)
        d = dict()
        d[self.terminal] = 5
        self.assertTrue(terminal_eq in d)
        self.assertFalse(terminal_ne_name in d)
        self.assertFalse(terminal_ne_type in d)
        d[terminal_eq] = 7
        self.assertTrue(d[self.terminal] == 7)

    def test_arity(self):
        self.assertEqual(self.terminal.arity, 3)
    
    def test_type(self):
        self.assertEqual(self.terminal.type, (self.nl1, self.nl2, self.nl3))
    


class TestNode(unittest.TestCase):
    
    def setUp(self):
        self.label = NodeLabel("label")
        self.node1 = Node(self.label)
        self.node2 = Node(self.label, id="id2")

    def test_id(self):
        self.assertEqual(self.node2.id, "id2")

    def test_bad_id(self):
        with self.assertRaises(TypeError):
            n = Node(self.label, id=1234)


class TestEdge(unittest.TestCase):
    
    def setUp(self):
        self.nl1   = NodeLabel("nl1")
        self.nl2   = NodeLabel("nl2")
        self.node1 = Node(self.nl1)
        self.node2 = Node(self.nl2)
        
        self.el1   = EdgeLabel("el1", (self.nl1, self.nl2), is_terminal=True)
        self.el2   = EdgeLabel("el2", (self.nl2,), is_terminal=True)
        
        self.edge1 = Edge(self.el1, (self.node1, self.node2))
        self.edge2 = Edge(self.el2, (self.node2,))
    
    def test_init_bad_input(self):
        # nodes have wrong labels
        with self.assertRaises(Exception):
            bad_edge = self.Edge(self.el2, (self.node1,))
        # list of nodes has wrong arity
        with self.assertRaises(Exception):
            bad_edge = self.Edge(self.el2, (self.node2, self.node2))

        # non-string id
        with self.assertRaises(TypeError):
            n = Edge(self.el1, (self.node1, self.node2), id=1234)
        
class TestGraph(unittest.TestCase):

    def setUp(self):
        self.nl1   = NodeLabel("nl1")
        self.nl2   = NodeLabel("nl2")
        self.node1 = Node(self.nl1, id='node1')
        self.node2 = Node(self.nl2)
        
        self.el1   = EdgeLabel("el1", (self.nl1, self.nl2), is_terminal=True)
        self.el2   = EdgeLabel("el2", (self.nl2,), is_nonterminal=True)
        self.edge1 = Edge(self.el1, (self.node1, self.node2), id='edge1')
        self.edge2 = Edge(self.el2, (self.node2,))
        
        self.graph = Graph()
        self.graph.add_node(self.node1)
        self.graph.add_node(self.node2)
        self.graph.add_edge(self.edge1)
        self.graph.add_edge(self.edge2)
        self.graph.ext = [self.node2]

    def test_add_node(self):
        nodes = self.graph.nodes()
        self.assertEqual(len(nodes), 2)
        self.assertTrue(self.node1 in nodes)
        self.assertTrue(self.node2 in nodes)

    def test_add_node_duplicate(self):
        # Can't add a Node already in a Graph
        with self.assertRaises(ValueError):
            self.graph.add_node(self.node1)
        # nor a copy of a Node already in a Graph
        new_node = Node(self.nl1, id=self.node1.id)
        with self.assertRaises(ValueError):
            self.graph.add_node(new_node)
    
    def test_add_edge(self):
        edges = self.graph.edges()
        self.assertEqual(len(edges), 2)
        self.assertTrue(self.edge1 in edges)
        self.assertTrue(self.edge2 in edges)

    def test_add_edge_duplicate(self):
        # Can't add an Edge already in a Graph
        with self.assertRaises(ValueError):
            self.graph.add_edge(self.edge1)
        # nor a copy of an Edge already in a Graph
        new_edge = Edge(self.el1, (self.node1, self.node2), id=self.edge1.id)
        with self.assertRaises(ValueError):
            self.graph.add_edge(new_edge)
        # nor an Edge with a conflicting label
        new_edge = Edge(EdgeLabel("el1", (), is_terminal=True), ())
        with self.assertRaises(ValueError):
            self.graph.add_edge(new_edge)
    
    def test_set_ext(self):
        ext = self.graph.ext
        self.assertEqual(ext, (self.node2,))
    
    def test_add_nodes_implicitly(self):
        node3 = Node(self.nl1)
        edge3 = Edge(self.el1, (node3, self.node2))
        self.graph.add_edge(edge3)
        self.assertTrue(node3 in self.graph.nodes())
        self.assertTrue(node3.id in self.graph._nodes.keys())
        
        node4 = Node(self.nl1)
        self.graph.ext = (self.node2, node4)
        self.assertTrue(node4 in self.graph.nodes())
        self.assertTrue(node4.id in self.graph._nodes.keys())

    def test_arity_and_type(self):
        self.assertEqual(self.graph.arity, 1)
        self.assertEqual(self.graph.type, (self.nl2,))

    def test_terminals_and_nonterminals(self):
        self.assertEqual(self.graph.terminals(), [self.edge1.label])
        self.assertEqual(self.graph.nonterminals(), [self.edge2.label])

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

        node4 = Node(self.nl1)
        self.graph.add_node(node4)
        self.graph.ext = [node4]
        with self.assertRaises(ValueError):
            self.graph.remove_node(node4) # because it's an external node

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
        
        copy = Graph()
        copy.add_node(self.node1)
        copy.add_node(self.node2)
        copy.add_edge(self.edge1)
        copy.add_edge(self.edge2)
        copy.ext = [self.node2]

        self.assertEqual(self.graph, copy)

    def test_convenience(self):
        g = Graph()
        node1 = g.new_node('nl1', id='node1')
        node2 = g.new_node('nl2')
        edge1 = g.new_edge('el1', [node1, node2], is_terminal=True, id='edge1')
        edge2 = g.new_edge('el2', [node2], is_nonterminal=True)
        g.ext = [node2]

        self.assertEqual(set(g.nodes()), {node1, node2})
        self.assertEqual(node1, self.node1)
        self.assertEqual(node2.label, self.node2.label)
        self.assertEqual(set(g.edges()), {edge1, edge2})
        self.assertEqual(edge1.label, self.edge1.label)
        self.assertEqual(edge1.label, self.edge1.label)
        self.assertEqual(edge1.nodes, (node1, node2))
        self.assertEqual(edge2.label, self.edge2.label)
        self.assertEqual(edge2.nodes, (node2,))
        self.assertEqual(g.ext, (node2,))

        
class TestHRGRule(unittest.TestCase):

    def setUp(self):
        nl = NodeLabel("nl1")
        
        node1 = Node(nl)
        node2 = Node(nl)

        terminal = EdgeLabel("terminal", (nl, nl), is_terminal=True)
        nonterminal_mismatch = EdgeLabel("nonterminal1", (nl,), is_nonterminal=True)
        nonterminal_match = EdgeLabel("nonterminal2", (nl, nl), is_nonterminal=True)
        
        graph = Graph()
        graph.add_node(node1)
        graph.add_node(node2)
        graph.ext = [node1, node2]
        
        with self.assertRaises(Exception):
            rule = HRGRule(terminal, graph)
        with self.assertRaises(Exception):
            rule = HRGRule(nonterminal_mismatch, graph)
        self.rule = HRGRule(nonterminal_match, graph)

    def test_copy_equal(self):
        rule = self.rule
        copy = self.rule.copy()
        self.assertNotEqual(id(rule), id(copy))
        self.assertEqual(rule, copy)

class TestHRG(unittest.TestCase):

    def setUp(self):
        self.nl1   = NodeLabel("nl1")
        self.nl2   = NodeLabel("nl2")
        self.node1 = Node(self.nl1)
        self.node2 = Node(self.nl2)
        
        self.el1   = EdgeLabel("el1", (self.nl1, self.nl2), is_terminal=True)
        self.el2   = EdgeLabel("el2", (self.nl2,), is_nonterminal=True)
        self.edge1 = Edge(self.el1, (self.node1, self.node2))
        self.edge2 = Edge(self.el2, (self.node2,))
        
        self.graph = Graph()
        self.graph.add_node(self.node1)
        self.graph.add_node(self.node2)
        self.graph.add_edge(self.edge1)
        self.graph.add_edge(self.edge2)
        self.graph.ext = []
        
        self.start = EdgeLabel("S", tuple(), is_nonterminal=True)
        self.rule = HRGRule(self.start, self.graph)
        
        self.node3 = Node(self.nl2)
        self.graph2 = Graph()
        self.graph2.add_node(self.node3)
        self.graph2.ext = [self.node3]
        self.rule2 = HRGRule(self.el2, self.graph2)

        self.hrg = HRG(self.start)
        self.hrg.add_node_label(self.nl1)
        self.hrg.add_node_label(self.nl2)
        self.hrg.add_edge_label(self.el1)
        self.hrg.add_edge_label(self.el2)
        self.hrg.add_edge_label(self.start)
        self.hrg.add_rule(self.rule)
        self.hrg.add_rule(self.rule2)

    def test_add_node_label(self):
        node_labels = self.hrg.node_labels()
        self.assertEqual(len(node_labels), 2)
        self.assertTrue(self.nl1 in node_labels)
        self.assertTrue(self.nl2 in node_labels)
        # add a node label which is a different object but
        # equivalent to an existing node label; code should
        # treat them as the same node label
        nl3 = NodeLabel("nl1")
        self.hrg.add_node_label(nl3)
        self.assertEqual(len(self.hrg.node_labels()), 2)
    
    def test_nonterminals(self):
        nonterminals = self.hrg.nonterminals()
        self.assertEqual(len(nonterminals), 2)
        self.assertTrue(self.start in nonterminals)
        self.assertTrue(self.el2 in nonterminals)
        
    def test_terminals(self):
        terminals = self.hrg.terminals()
        self.assertEqual(len(terminals), 1)
        self.assertTrue(self.el1 in terminals)
    
    def test_add_edge_label_bad_input(self):
        # conflicting edge label names
        # nonterminal with nonterminal
        nt = EdgeLabel("el2", (self.nl1, self.nl1), is_nonterminal=True)
        with self.assertRaises(Exception):
            self.hrg.add_edge_label(nt)
        
        # terminal with terminal
        t = EdgeLabel("el1", (self.nl1, self.nl1), is_terminal=True)
        with self.assertRaises(Exception):
            self.hrg.add_edge_label(t)
        
        # nonterminal with terminal
        nt = EdgeLabel("el1", (self.nl1, self.nl1), is_nonterminal=True)
        with self.assertRaises(Exception):
            self.hrg.add_edge_label(nt)
        
        # terminal with nonterminal
        t = EdgeLabel("el2", (self.nl1, self.nl1), is_nonterminal=True)
        with self.assertRaises(Exception):
            self.hrg.add_edge_label(t)
            
        # it should allow you to add the same edge label a second time
        self.hrg.add_edge_label(self.start)
        self.hrg.add_edge_label(self.el1)

    def test_set_start(self):
        self.assertEqual(self.hrg.start, self.start)

    def test_add_rule(self):
        all_rules = self.hrg.all_rules()
        self.assertEqual(len(all_rules), 2)
        self.assertTrue(self.rule in all_rules)
        self.assertTrue(self.rule2 in all_rules)
        
        start_rules = self.hrg.rules(self.start)
        self.assertEqual(len(start_rules), 1)
        self.assertTrue(self.rule in start_rules)

    def test_implicitly_add_node_and_edge_labels(self):
        new_nl = NodeLabel("nl3")
        new_nt = EdgeLabel("nt1", (new_nl, new_nl), is_nonterminal=True)
        new_t  = EdgeLabel("nt2", (new_nl,), is_terminal=True)
        
        new_node1 = Node(new_nl)
        new_node2 = Node(new_nl)
        new_edge  = Edge(new_t, (new_node1,))
        new_graph = Graph()
        new_graph.add_node(new_node1)
        new_graph.add_node(new_node2)
        new_graph.add_edge(new_edge)
        new_graph.ext = (new_node1, new_node2)
        new_rule  = HRGRule(new_nt, new_graph)
        
        self.hrg.add_rule(new_rule)
        
        self.assertTrue(new_nl in self.hrg.node_labels())
        self.assertTrue(new_nt in self.hrg.nonterminals())
        self.assertTrue(new_t  in self.hrg.terminals())
        
    def test_copy_equal(self):
        hrg = self.hrg
        copy = self.hrg.copy()
        self.assertNotEqual(id(hrg), id(copy))
        self.assertEqual(hrg, copy)

    def test_convenience(self):
        copy = HRG(self.hrg.start.name)
        for rule in self.hrg.all_rules():
            copy.new_rule(rule.lhs.name, rule.rhs)
        self.assertEqual(self.hrg, copy)

class TestFactorGraph(unittest.TestCase):

    def setUp(self):
        TestGraph.setUp(self)
        self.fg = FactorGraph.from_graph(self.graph)
        
        self.dom1 = FiniteDomain(['foo', 'bar', 'baz'])
        self.dom2 = FiniteDomain(['jia', 'yi', 'bing', 'ding'])

        self.fac1 = FiniteFactor([self.dom1, self.dom2],
                                      [[1, 2, 3, 4],
                                       [5, 6, 7, 8],
                                       [1, 2, 3, 4]])
        self.fac2 = FiniteFactor([self.dom2], [0, 0, 0, 0])
        
    def test_shape(self):
        fg = self.fg
        fg.add_domain(self.nl1, self.dom1)
        fg.add_domain(self.nl2, self.dom2)
        # Test all formats of input to the function
        self.assertEqual(fg.shape([self.nl1, self.nl1, self.nl2]), (3, 3, 4))
        self.assertEqual(fg.shape([self.node2, self.node1, self.node2]), (4, 3, 4))
        self.assertEqual(fg.shape(self.el1), (3, 4))
        self.assertEqual(fg.shape(self.edge2), (4,))
        # Test that empty input has correct shape
        self.assertEqual(fg.shape([]), tuple())

    def test_convenience(self):
        fg = self.fg
        fg.new_finite_domain(self.nl1.name, self.dom1.values)
        self.assertTrue(fg.domains[self.nl1.name] == self.dom1)
        fg.new_finite_domain(self.nl2.name, self.dom2.values)
        self.assertTrue(fg.domains[self.nl2.name] == self.dom2)
        fg.new_finite_factor(self.el1.name, self.fac1.weights)
        self.assertTrue(fg.factors[self.el1.name] == self.fac1)

class TestFGG(unittest.TestCase):

    def setUp(self):
        TestHRG.setUp(self)
        self.fgg = FGG.from_hrg(self.hrg)
        
        self.dom1 = FiniteDomain(['foo', 'bar', 'baz'])
        self.dom2 = FiniteDomain(['jia', 'yi', 'bing', 'ding'])

        self.fac1 = FiniteFactor([self.dom1, self.dom2],
                                      [[1, 2, 3, 4],
                                       [5, 6, 7, 8],
                                       [1, 2, 3, 4]])
        self.fac2 = FiniteFactor([self.dom2], [0, 0, 0, 0])


    def test_shape(self):
        fgg = self.fgg
        fgg.add_domain(self.nl1, self.dom1)
        fgg.add_domain(self.nl2, self.dom2)
        # Test all formats of input to the function
        self.assertEqual(fgg.shape([self.nl1, self.nl1, self.nl2]), (3, 3, 4))
        self.assertEqual(fgg.shape([self.node2, self.node1, self.node2]), (4, 3, 4))
        self.assertEqual(fgg.shape(self.el1), (3, 4))
        self.assertEqual(fgg.shape(self.edge2), (4,))
        # Test that empty input has correct shape
        self.assertEqual(fgg.shape([]), tuple())

    def test_convenience(self):
        fgg = self.fgg
        fgg.new_finite_domain(self.nl1.name, self.dom1.values)
        self.assertTrue(fgg.domains[self.nl1.name] == self.dom1)
        fgg.new_finite_domain(self.nl2.name, self.dom2.values)
        self.assertTrue(fgg.domains[self.nl2.name] == self.dom2)
        fgg.new_finite_factor(self.el1.name, self.fac1.weights)
        self.assertTrue(fgg.factors[self.el1.name] == self.fac1)

        
if __name__ == "__main__":
    unittest.main()
