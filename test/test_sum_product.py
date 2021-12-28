from fggs import sum_product, FGG, Interpretation, CategoricalFactor
from fggs.sum_product import scc, MultiTensor, SumProduct
from fggs import FGG, json_to_hrg, json_to_interp, json_to_fgg
import unittest, warnings, torch, random, json

def load_json(filename):
    with open(filename) as f:
        return json.load(f)

class TestSumProduct(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings('ignore', message='.*maximum iteration.*')
        self.fgg_1 = json_to_fgg(load_json('test/hmm.json'))
        self.fgg_2 = json_to_fgg(load_json('test/example12p.json'))
        self.fgg_3 = json_to_fgg(load_json('test/simplefgg.json'))
        self.fgg_4 = json_to_fgg(load_json('test/barhillel.json'))
        self.fgg_5 = json_to_fgg(load_json('test/linear.json'))

    def test_fixed_point_1(self):
        self.assertAlmostEqual(sum_product(self.fgg_1, method='fixed-point').item(), 1.0, places=2)

    def test_fixed_point_2(self):
        from math import sqrt
        def exact_value(p):
            # minimal solution of (x, y) = (2pxy + (1 - p), p(x^2 + y^2)) where x = p(true) and y = p(false)
            return ((3 - 2*p - sqrt(1 + 4*p - 4*p**2))/(4*p), ( 1 - 2*p + sqrt(1 + 4*p - 4*p**2))/(4*p)) if p > 0.5 \
              else ((1 + 2*p - sqrt(1 + 4*p - 4*p**2))/(4*p), (-1 + 2*p + sqrt(1 + 4*p - 4*p**2))/(4*p))
        for p in (random.uniform(0.01, 0.99) for _ in range(10)):
            self.fgg_2.interp.factors[self.fgg_2.grammar.get_edge_label('p')].weights = [1 - p, p]
            for A, B in zip(sum_product(self.fgg_2, method='fixed-point'), exact_value(p)):
                self.assertAlmostEqual(A.item(), B, places=2)

    def test_fixed_point_3(self):
        self.assertAlmostEqual(sum_product(self.fgg_3, method='fixed-point').item(), 0.25, places=2)

    def test_fixed_point_5(self):
        self.assertAlmostEqual(sum_product(self.fgg_5, method='fixed-point').item(), 7.5, places=2)

    def xtest_broyden_1(self):
        self.assertAlmostEqual(sum_product(self.fgg_1, method='broyden').item(), 1.0, places=2)

    def xtest_broyden_2(self):
        from math import sqrt
        def exact_value(p):
            # minimal solution of (x, y) = (2pxy + (1 - p), p(x^2 + y^2)) where x = p(true) and y = p(false)
            return ((3 - 2*p - sqrt(1 + 4*p - 4*p**2))/(4*p), ( 1 - 2*p + sqrt(1 + 4*p - 4*p**2))/(4*p)) if p > 0.5 \
              else ((1 + 2*p - sqrt(1 + 4*p - 4*p**2))/(4*p), (-1 + 2*p + sqrt(1 + 4*p - 4*p**2))/(4*p))
        for p in (random.uniform(0.01, 0.99) for _ in range(10)):
            self.fgg_2.interp.factors[self.fgg_2.grammar.get_edge_label('p')].weights = [1 - p, p]
            for A, B in zip(sum_product(self.fgg_2, method='broyden'), exact_value(p)):
                self.assertAlmostEqual(A.item(), B, places=2)

    def test_newton_3(self):
        self.assertAlmostEqual(sum_product(self.fgg_3, method='newton').item(), 0.25, places=2)

    def test_autograd(self):
        import torch
        torch.set_default_dtype(torch.double)
        torch.autograd.set_detect_anomaly(True)
        for fgg in [self.fgg_5]:
            in_labels = list(fgg.interp.factors.keys())
            in_values = [torch.tensor(fac.weights, requires_grad=True, dtype=torch.get_default_dtype())
                         for fac in fgg.interp.factors.values()]
            out_labels = list(fgg.grammar.nonterminals())
            def f(*in_values):
                opts = {'method': 'fixed-point', 'tol': 1e-6, 'kmax': 1000}
                return SumProduct.apply(fgg, opts, in_labels, out_labels, *in_values)
            self.assertTrue(torch.autograd.gradcheck(f, in_values))

    def test_disconnected_node(self):
        fgg = json_to_fgg(load_json('test/disconnected_node.json'))
        self.assertAlmostEqual(sum_product(fgg, method='fixed-point').sum().item(), 54.)
        self.assertAlmostEqual(sum_product(fgg, method='newton').sum().item(), 54.) # disabled until J uses sum_product_edges
        self.assertAlmostEqual(sum_product(fgg, method='linear').sum().item(), 54.)

    def test_4(self):
        z_fp = sum_product(self.fgg_4, method='fixed-point')
        z_newton = sum_product(self.fgg_4, method='newton')
        self.assertAlmostEqual(torch.norm(z_fp - z_newton).item(), 0., places=2)

    def test_linear_1(self):
        self.assertAlmostEqual(sum_product(self.fgg_1, method='linear').item(), 1.0, places=2)
    def test_linear_5(self):
        self.assertAlmostEqual(sum_product(self.fgg_5, method='linear').item(), 7.5, places=2)
        
    def xtest_linear_1_grad(self):
        import fggs
        interp = Interpretation()
        for nl, dom in self.fgg_1.interp.domains.items():
            interp.add_domain(nl, dom)
        for el, fac in self.fgg_1.interp.factors.items():
            fac = CategoricalFactor(fac.domains, fac.weights)
            fac.weights = torch.tensor(fac.weights, requires_grad=True, dtype=torch.get_default_dtype())
            interp.add_factor(el, fac)
        fgg = FGG(self.fgg_1.grammar, interp)
        z = sum_product(fgg, method='linear')
        z.backward()
        # As long as there's no error, the gradient should be correct


class TestSCC(unittest.TestCase):
    def test_scc(self):
        g = json_to_fgg(load_json('test/hmm.json')).grammar
        self.assertEqual(scc(g), [{g.get_edge_label('X')}, {g.get_edge_label('S')}])

class TestMultiTensor(unittest.TestCase):
    def setUp(self):
        def load(filename):
            with open(filename) as f:
                return json.load(f)
        self.fgg_1 = json_to_fgg(load('test/hmm.json'))
        self.S, self.X = self.fgg_1.grammar.get_edge_label('S'), self.fgg_1.grammar.get_edge_label('X')
        
    def test_basic(self):
        mt = MultiTensor.initialize(self.fgg_1)
        self.assertEqual(list(mt.size()), [7])
        self.assertEqual(list(mt.dict[self.S].size()), [])
        self.assertEqual(list(mt.dict[self.X].size()), [6])

    def test_square(self):
        mt = MultiTensor.initialize(self.fgg_1, ndim=2)
        self.assertEqual(list(mt.size()), [7, 7])
        self.assertEqual(list(mt.dict[self.S, self.S].size()), [])
        self.assertEqual(list(mt.dict[self.S, self.X].size()), [6])
        self.assertEqual(list(mt.dict[self.X, self.S].size()), [6])
        self.assertEqual(list(mt.dict[self.X, self.X].size()), [6, 6])
        
if __name__ == '__main__':
    unittest.main()
