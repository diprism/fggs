from fggs import sum_product, FGG, Interpretation, CategoricalFactor
from fggs.sum_product import scc
from fggs import FGG, json_to_hrg, json_to_interp, json_to_fgg
import unittest, warnings, torch, random, json

class TestSumProduct(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings('ignore', message='.*maximum iteration.*')
        def load(filename):
            with open(filename) as f:
                return json.load(f)
        self.fgg_1 = json_to_fgg(load('test/hmm.json'))
        self.fgg_2 = json_to_fgg(load('test/example12p.json'))
        self.fgg_3 = json_to_fgg(load('test/simplefgg.json'))
        self.fgg_4 = json_to_fgg(load('test/barhillel.json'))

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

    def test_broyden_1(self):
        self.assertAlmostEqual(sum_product(self.fgg_1, method='broyden').item(), 1.0, places=2)

    def test_broyden_2(self):
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

    def xtest_4(self):
        z_fp = sum_product(self.fgg_4, method='fixed-point')
        z_newton = sum_product(self.fgg_4, method='newton')
        self.assertAlmostEqual(torch.norm(z_fp - z_newton), 0., places=2)

    def test_linear_1(self):
        self.assertAlmostEqual(sum_product(self.fgg_1, method='linear').item(), 1.0, places=2)
        
    def test_linear_1_grad(self):
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
        with open('test/hmm.json') as f:
            g = json_to_hrg(json.load(f)['grammar'])
        self.assertEqual(scc(g), [{g.get_edge_label('X')}, {g.get_edge_label('S')}])

if __name__ == '__main__':
    unittest.main()
