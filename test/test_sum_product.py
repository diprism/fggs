from fggs import sum_product
from fggs.sum_product import scc
from fggs import FGG, json_to_hrg, json_to_interp
import unittest, warnings, random, json

def load_json(filename):
    with open(filename) as f:
        return json.load(f)

def load_fgg(gfilename, ifilename):
    return FGG(json_to_hrg(load_json(gfilename)),
               json_to_interp(load_json(ifilename)))
    
class TestSumProduct(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings('ignore', message='.*maximum iteration.*')
        self.fgg_1 = load_fgg('test/hmm.json', 'test/hmm_interp.json')
        self.fgg_2 = load_fgg('test/example12p.json', 'test/example12p_interp.json')
        self.fgg_3 = load_fgg('test/simplefgg.json', 'test/simplefgg_interp.json')

    def test_fixed_point_1(self):
        self.assertAlmostEqual(sum_product(self.fgg_1, method='fixed-point').item(), 1.0, places=2)

    def test_fixed_point_2(self):
        from math import sqrt
        def exact_value(p):
            # minimal solution of (x, y) = (2pxy + (1 - p), p(x^2 + y^2)) where x = p(true) and y = p(false)
            return ((3 - 2*p - sqrt(1 + 4*p - 4*p**2))/(4*p), ( 1 - 2*p + sqrt(1 + 4*p - 4*p**2))/(4*p)) if p > 0.5 \
              else ((1 + 2*p - sqrt(1 + 4*p - 4*p**2))/(4*p), (-1 + 2*p + sqrt(1 + 4*p - 4*p**2))/(4*p))
        for p in (random.uniform(0.01, 0.99) for _ in range(10)):
            self.fgg_2.interp.factors[self.fgg_2.grammar.get_edge_label('p')]._weights = [1 - p, p]
            for A, B in zip(sum_product(self.fgg_2, method='fixed-point'), exact_value(p)):
                self.assertAlmostEqual(A.item(), B, places=2)

    def test_broyden_1(self):
        self.assertAlmostEqual(sum_product(self.fgg_1, method='broyden').item(), 1.0, places=2)

    def test_broyden_2(self):
        from math import sqrt
        def exact_value(p):
            # minimal solution of (x, y) = (2pxy + (1 - p), p(x^2 + y^2)) where x = p(true) and y = p(false)
            return ((3 - 2*p - sqrt(1 + 4*p - 4*p**2))/(4*p), ( 1 - 2*p + sqrt(1 + 4*p - 4*p**2))/(4*p)) if p > 0.5 \
              else ((1 + 2*p - sqrt(1 + 4*p - 4*p**2))/(4*p), (-1 + 2*p + sqrt(1 + 4*p - 4*p**2))/(4*p))
        for p in (random.uniform(0.01, 0.99) for _ in range(10)):
            self.fgg_2.interp.factors[self.fgg_2.grammar.get_edge_label('p')]._weights = [1 - p, p]
            for A, B in zip(sum_product(self.fgg_2, method='broyden'), exact_value(p)):
                self.assertAlmostEqual(A.item(), B, places=2)

    def test_newton_3(self):
        self.assertAlmostEqual(sum_product(self.fgg_3, method='newton').item(), 0.25, places=2)

    def test_disconnected_node(self):
        fgg = load_fgg('test/disconnected_node.json', 'test/disconnected_node_interp.json')
        self.assertAlmostEqual(sum_product(fgg, method='fixed-point').item(), 6.)
        self.assertAlmostEqual(sum_product(fgg, method='newton').item(), 6.)

class TestSCC(unittest.TestCase):
    def test_scc(self):
        g = json_to_hrg(load_json('test/hmm.json'))
        self.assertEqual(scc(g), [{g.get_edge_label('X')}, {g.get_edge_label('S')}])
        
if __name__ == '__main__':
    unittest.main()
