import unittest, json
from fggs import *
from fggs.utils import naive_graph_isomorphism

def load_fgg(filename):
    with open(filename) as f:
        return json_to_fgg(json.load(f))

class TestDerivation(unittest.TestCase):
    def setUp(self):
        self.fgg = load_fgg('test/hmm.json')

    def test_derive(self):
        S = self.fgg.get_edge_label('S')
        X = self.fgg.get_edge_label('X')
        rule_S = self.fgg.rules(S)[0]
        rule_X1 = self.fgg.rules(X)[0]
        rule_X2 = self.fgg.rules(X)[1]
        deriv = FGGDerivation(
            self.fgg,
            rule_S,
            {rule_S.rhs._nodes['v0']: 4}, # BOS
            {
                rule_S.rhs._edges['e1']: FGGDerivation(
                    self.fgg,
                    rule_X1,
                    {
                        rule_X1.rhs._nodes['v0']: 4, # BOS
                        rule_X1.rhs._nodes['v1']: 0, # DT
                        rule_X1.rhs._nodes['v2']: 0, # the
                    },
                    {
                        rule_X1.rhs._edges['e2']: FGGDerivation(
                            self.fgg,
                            rule_X2,
                            {
                                rule_X1.rhs._nodes['v0']: 0, # the
                                rule_X1.rhs._nodes['v1']: 5, # EOS
                            },
                            {}
                        )
                    }
                )
            }
        )

        g = Graph()
        bos = g.new_node('T')
        dt  = g.new_node('T')
        the = g.new_node('W')
        eos = g.new_node('T')
        g.new_edge('is_bos',     [bos],      is_terminal=True)
        g.new_edge('transition', [bos, dt],  is_terminal=True)
        g.new_edge('transition', [dt,  eos], is_terminal=True)
        g.new_edge('emission',   [dt,  the], is_terminal=True)
        g.new_edge('is_eos',     [eos],      is_terminal=True)

        tdom = self.fgg.domains['T']
        wdom = self.fgg.domains['W']
        
        a = {}
        a[bos] = tdom.numberize('BOS')
        a[dt]  = tdom.numberize('DT')
        a[the] = wdom.numberize('the')
        a[eos] = tdom.numberize('EOS')

        derived, asst = deriv.derive()
        result, m = naive_graph_isomorphism(derived, g)

        for node in asst:
            self.assertEqual(asst[node], a[m[node]])
        
        self.assertTrue(result, m)

class TestReplace(unittest.TestCase):
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
        self.graph.ext = [self.node2]

    def test_replace(self):
        g = self.graph.copy()
        (node_map, edge_map) = replace_edge(g, self.edge2, self.graph)
        self.assertEqual(sorted(v.label.name for v in g.nodes()), ['nl1', 'nl1', 'nl2'])
        self.assertEqual(sorted(e.label.name for e in g.edges()), ['el1', 'el1', 'el2'])

class TestStartGraph(unittest.TestCase):        
    def setUp(self):
        self.start = EdgeLabel("S", tuple(), is_nonterminal=True)
        self.hrg = HRG(self.start)

    def test_start_graph(self):
        s = self.hrg.start
        g = start_graph(self.hrg)
        self.assertEqual([e.label for e in g.edges()], [s])
        self.assertEqual(len(g.nodes()), len(s.type))
    
if __name__ == '__main__':
    unittest.main()
