from fggs import sum_product, FGG, Interpretation, CategoricalFactor, json_to_fgg
from fggs.sum_product import scc, MultiTensor
import unittest, warnings, torch, random, json

def load_fgg(filename):
    with open(filename) as f:
        return json_to_fgg(json.load(f))

class Example:
    def __init__(self, filename, linear=False, clean=True, value=None):
        self.name = filename
        self.linear = linear
        self.clean = clean
        if value is not None:
            self.value = value
        self.fgg = load_fgg(filename)

    def exact(self):
        return self.value

    def __str__(self):
        return self.name

class PPLExample(Example):
    def __init__(self, p):
        super().__init__('test/example12p.json', clean=False)
        self.p = p
        self.fgg.interp.factors[self.fgg.grammar.get_edge_label('p')].weights = [1 - p, p]
        
    def exact(self):
        from math import sqrt
        p = self.p
        # minimal solution of (x, y) = (2pxy + (1 - p), p(x^2 + y^2)) where x = p(true) and y = p(false)
        if p > 0.5:
            return torch.tensor([(3 - 2*p - sqrt(1 + 4*p - 4*p**2))/(4*p), ( 1 - 2*p + sqrt(1 + 4*p - 4*p**2))/(4*p)])
        else:
            return torch.tensor([(1 + 2*p - sqrt(1 + 4*p - 4*p**2))/(4*p), (-1 + 2*p + sqrt(1 + 4*p - 4*p**2))/(4*p)])
    
    def __str__(self):
        return f'{self.name} (p={self.p})'

class TestSumProduct(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings('ignore', message='.*maximum iteration.*')

        self.examples = [
            Example('test/hmm.json', value=1., linear=True),
            PPLExample(0.1), PPLExample(0.5), PPLExample(0.9),
            Example('test/simplefgg.json', value=0.25),
            Example('test/disconnected_node.json', value=torch.tensor([18., 18., 18.]), linear=True),
            Example('test/barhillel.json', value=torch.tensor([[0.1129, 0.0129], [0.0129, 0.1129]])),
            Example('test/test.json', value=torch.tensor([7., 1., 1.]), clean=False),
        ]
        

    def test_fixed_point(self):
        for example in self.examples:
            with self.subTest(example=str(example)):
                z = sum_product(example.fgg, method='fixed-point')
                z_exact = example.exact()
                self.assertTrue(torch.norm(z - z_exact) < 1e-2,
                                f'{z} != {z_exact}')

    def test_linear(self):
        for example in self.examples:
            with self.subTest(example=str(example)):
                if example.linear:
                    z = sum_product(example.fgg, method='linear')
                    z_exact = example.exact()
                    self.assertTrue(torch.norm(z - z_exact) < 1e-10,
                                    f'{z} != {z_exact}')
                else:
                    with self.assertRaises(ValueError):
                        _ = sum_product(example.fgg, method='linear')

    def test_linear_grad(self):
        for example in self.examples:
            if not example.linear: continue
            with self.subTest(example=str(example)):
                interp = Interpretation()
                for nl, dom in example.fgg.interp.domains.items():
                    interp.add_domain(nl, dom)
                for el, fac in example.fgg.interp.factors.items():
                    fac = CategoricalFactor(fac.domains, fac.weights)
                    fac.weights = torch.tensor(fac.weights, requires_grad=True, dtype=torch.get_default_dtype())
                    interp.add_factor(el, fac)
                fgg = FGG(example.fgg.grammar, interp)
                z = sum_product(fgg, method='linear').sum()
                z.backward()

    def test_newton(self):
        for example in self.examples:
            if not example.clean: continue # not implemented yet
            with self.subTest(example=str(example)):
                z = sum_product(example.fgg, method='newton')
                z_exact = example.exact()
                self.assertTrue(torch.norm(z - z_exact) < 1e-2,
                                f'{z} != {z_exact}')
                
    def test_broyden(self):
        for example in self.examples:
            with self.subTest(example=str(example)):
                z = sum_product(example.fgg, method='broyden')
                z_exact = example.exact()
                self.assertTrue(torch.norm(z - z_exact) < 1e-2,
                                f'{z} != {z_exact}')

                
class TestSCC(unittest.TestCase):
    def test_scc(self):
        g = load_fgg('test/hmm.json').grammar
        self.assertEqual(scc(g), [{g.get_edge_label('X')}, {g.get_edge_label('S')}])

class TestMultiTensor(unittest.TestCase):
    def setUp(self):
        self.fgg_1 = load_fgg('test/hmm.json')
        
    def test_basic(self):
        fgg = self.fgg_1
        hrg = fgg.grammar
        mt = MultiTensor.initialize(fgg)
        self.assertEqual(list(mt.size()), [7])
        self.assertEqual(list(mt.get(hrg.get_edge_label('S')).size()), [])
        self.assertEqual(list(mt.get(hrg.get_edge_label('X')).size()), [6])

    def test_square(self):
        fgg = self.fgg_1
        hrg = fgg.grammar
        mt = MultiTensor.initialize(self.fgg_1, ndim=2)
        self.assertEqual(list(mt.size()), [7, 7])
        S, X = hrg.get_edge_label('S'), hrg.get_edge_label('X')
        self.assertEqual(list(mt.get(S, S).size()), [])
        self.assertEqual(list(mt.get(S, X).size()), [6])
        self.assertEqual(list(mt.get(X, S).size()), [6])
        self.assertEqual(list(mt.get(X, X).size()), [6, 6])
        
if __name__ == '__main__':
    unittest.main()
