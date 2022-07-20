import unittest
from fggs.domains import *
from fggs.factors import *
from fggs.fggs import *
from fggs.utils import *

def load_fgg(filename):
    import json
    from fggs import json_to_fgg
    with open(filename) as f:
        return json_to_fgg(json.load(f))

class TestUniqueName(unittest.TestCase):
    
    def setUp(self):
        self.el1 = EdgeLabel("e", [], is_terminal=True)
        self.el2 = EdgeLabel("e_1", [], is_terminal=True)
        
    def test_unique_label_name(self):
        self.assertEqual(unique_label_name("e", [self.el1, self.el2]), "e_2")
        self.assertEqual(unique_label_name("f", [self.el1, self.el2]), "f")


class TestSingleton(unittest.TestCase):

    def setUp(self):
        self.nl1   = NodeLabel("nl1")
        self.node1 = Node(self.nl1)
        self.node2 = Node(self.nl1)
        
        self.el1   = EdgeLabel("el1", (self.nl1, self.nl1), is_terminal=True)
        self.edge1 = Edge(self.el1, (self.node1, self.node2))
        
        self.graph = Graph()
        self.graph.add_node(self.node1)
        self.graph.add_node(self.node2)
        self.graph.add_edge(self.edge1)
        self.graph.ext = [self.node1]
        
        self.start = EdgeLabel("<S>", [self.nl1], is_nonterminal=True)
        
    def test_singleton_hrg(self):
        g = singleton_hrg(self.graph)
        self.assertCountEqual(g.node_labels(), [self.nl1])
        self.assertCountEqual(g.edge_labels(), [self.el1, self.start])
        self.assertEqual(g.start, self.start)
        self.assertEqual(len(g.all_rules()), 1)
    
    def test_unique_start_name(self):
        s1_lab  = EdgeLabel("<S>", [], is_terminal=True)
        s1_edge = Edge(s1_lab, [])
        self.graph.add_edge(s1_edge)
        
        g = singleton_hrg(self.graph)
        self.assertEqual(g.start.name, "<S>_1")

        
class TestSCC(unittest.TestCase):
    def test_scc(self):
        g = load_fgg('test/hmm.json')
        self.assertEqual(scc(nonterminal_graph(g)), [{g.get_edge_label('X')}, {g.get_edge_label('S')}])
