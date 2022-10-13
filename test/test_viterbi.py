from fggs import FGG, Graph, ViterbiSemiring, json_to_fgg
from fggs.viterbi import viterbi
from fggs.utils import naive_graph_isomorphism
import unittest, torch, json, copy

def load_fgg(filename):
    with open(filename) as f:
        return json_to_fgg(json.load(f))

class TestSumProduct(unittest.TestCase):

    def setUp(self):
        self.fgg = load_fgg('test/hmm.json')
        self.graph = g = Graph()
        bos = g.new_node('T')
        dt  = g.new_node('T')
        the = g.new_node('W')
        nn  = g.new_node('T')
        cat = g.new_node('W')
        eos = g.new_node('T')
        g.new_edge('is_bos',     [bos],      is_terminal=True)
        g.new_edge('transition', [bos, dt],  is_terminal=True)
        g.new_edge('transition', [dt,  nn],  is_terminal=True)
        g.new_edge('transition', [nn,  eos], is_terminal=True)
        g.new_edge('emission',   [dt,  the], is_terminal=True)
        g.new_edge('emission',   [nn,  cat], is_terminal=True)
        g.new_edge('is_eos',     [eos],      is_terminal=True)

        tdom = self.fgg.domains['T']
        wdom = self.fgg.domains['W']
        
        self.asst = a = {}
        a[bos] = tdom.numberize('BOS')
        a[dt]  = tdom.numberize('DT')
        a[the] = wdom.numberize('the')
        a[nn]  = tdom.numberize('NN')
        a[cat] = wdom.numberize('cat')
        a[eos] = tdom.numberize('EOS')
        
    def test_viterbi(self):
        fgg = self.fgg.copy()
        for fac in fgg.factors.values():
            fac.weights = torch.log(torch.as_tensor(fac.weights))
        (facgraph, asst) = viterbi(fgg, ()).derive()
        result, m = naive_graph_isomorphism(facgraph, self.graph)
        self.assertTrue(result, m)
        for node in asst:
            self.assertEqual(asst[node], self.asst[m[node]])
                
if __name__ == '__main__':
    unittest.main()
