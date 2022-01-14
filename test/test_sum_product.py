from fggs import sum_product, FGG, Interpretation, CategoricalFactor, json_to_fgg
from fggs.sum_product import scc, MultiTensor, SumProduct
from fggs.semirings import *
import unittest, warnings, torch, random, json, copy

def load_fgg(filename):
    with open(filename) as f:
        return json_to_fgg(json.load(f))

class Example:
    def __init__(self, filename, linear=False, clean=True, slow=False, value=None):
        self.name = filename
        self.linear = linear
        self.clean = clean
        self.slow = slow
        if value is not None:
            self.value = value
        self.fgg = load_fgg(filename)

    def exact(self):
        return self.value

    def __str__(self):
        return self.name

class PPLExample(Example):
    def __init__(self, p):
        super().__init__('test/example12p.json', clean=False, slow=True)
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
            Example('test/hmm.json', value=1., linear=True, clean=False),
            PPLExample(0.1), PPLExample(0.5), PPLExample(0.9),
            Example('test/simplefgg.json', value=0.25),
            Example('test/disconnected_node.json', value=torch.tensor([18., 18., 18.]), linear=True),
            Example('test/barhillel.json', value=torch.tensor([[0.1129, 0.0129], [0.0129, 0.1129]])),
            Example('test/test.json', value=torch.tensor([7., 1., 1.]), clean=False),
            Example('test/linear.json', value=7.5, linear=True),
        ]

        for ex in self.examples:
            for fac in ex.fgg.interp.factors.values():
                if not isinstance(fac.weights, torch.Tensor):
                    fac.weights = torch.tensor(fac.weights, dtype=torch.get_default_dtype())


    def test_autograd(self):
        torch.set_default_dtype(torch.double)
        for example in self.examples:
            if example.slow: continue
            if not example.clean: continue # not implemented yet
            with self.subTest(example=str(example)):
                fgg = example.fgg
                in_labels = list(fgg.interp.factors.keys())
                in_values = [fac.weights.to(torch.double).requires_grad_(True)
                             for fac in fgg.interp.factors.values()]
                out_labels = list(fgg.grammar.nonterminals())
                def f(*in_values):
                    opts = {'method': 'fixed-point', 'tol': 1e-6, 'kmax': 100}
                    return SumProduct.apply(fgg, opts, in_labels, out_labels, *in_values)
                self.assertTrue(torch.autograd.gradcheck(f, in_values, atol=1e-3))

    def test_autograd_log(self):
        torch.set_default_dtype(torch.double)
        for example in self.examples:
            if example.slow: continue
            if not example.clean: continue # not implemented yet
            with self.subTest(example=str(example)):
                fgg = example.fgg
                in_labels = list(fgg.interp.factors.keys())
                in_values = [fac.weights.log().to(torch.double).requires_grad_(True)
                             for fac in fgg.interp.factors.values()]
                out_labels = list(fgg.grammar.nonterminals())
                def f(*in_values):
                    opts = {'method': 'fixed-point', 'tol': 1e-6, 'kmax': 100, 'semiring': LogSemiring}
                    ret = SumProduct.apply(fgg, opts, in_labels, out_labels, *in_values)
                    # put exp inside f to avoid gradcheck computing -inf - -inf
                    return tuple(torch.exp(z) for z in ret)
                self.assertTrue(torch.autograd.gradcheck(f, in_values, atol=1e-3))

    def test_fixed_point(self):
        for example in self.examples:
            with self.subTest(example=str(example)):
                z = sum_product(example.fgg, method='fixed-point')
                z_exact = example.exact()
                self.assertTrue(torch.norm(z - z_exact) < 1e-2,
                                f'{z} != {z_exact}')

    def test_fixed_point_log(self):
        for example in self.examples:
            with self.subTest(example=str(example)):
                interp = copy.deepcopy(example.fgg.interp)
                for fac in interp.factors.values():
                    fac.weights = torch.log(fac.weights)
                fgg = FGG(example.fgg.grammar, interp)
                z = torch.exp(sum_product(fgg, method='fixed-point', semiring=LogSemiring))
                z_exact = example.exact()
                self.assertTrue(torch.norm(z - z_exact) < 1e-2,
                                f'{z} != {z_exact}')

    def test_fixed_point_bool(self):
        for example in self.examples:
            with self.subTest(example=str(example)):
                interp = copy.deepcopy(example.fgg.interp)
                for fac in interp.factors.values():
                    fac.weights = fac.weights > 0.
                fgg = FGG(example.fgg.grammar, interp)
                z = sum_product(fgg, method='fixed-point', semiring=BoolSemiring, tol=0)
                z_exact = example.exact() > 0.
                self.assertTrue(torch.all(z == z_exact),
                                f'{z} != {z_exact}')

    def test_linear(self):
        for example in self.examples:
            if not example.clean: continue # not implemented yet
            with self.subTest(example=str(example)):
                if example.linear:
                    z = sum_product(example.fgg, method='linear')
                    z_exact = example.exact()
                    self.assertTrue(torch.norm(z - z_exact) < 1e-10,
                                    f'{z} != {z_exact}')
                else:
                    with self.assertRaises(ValueError):
                        _ = sum_product(example.fgg, method='linear')

    def test_newton(self):
        for example in self.examples:
            if not example.clean: continue # not implemented yet
            with self.subTest(example=str(example)):
                z = sum_product(example.fgg, method='newton')
                z_exact = example.exact()
                self.assertTrue(torch.norm(z - z_exact) < 1e-2,
                                f'{z} != {z_exact}')
                
    def xtest_broyden(self):
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

    def test_ops(self):
        x = MultiTensor.initialize(self.fgg_1)
        y = x + 1.
        self.assertTrue(torch.norm(y - (x._t + 1.)) < 1e-6)
        
if __name__ == '__main__':
    unittest.main()
