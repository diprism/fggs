from fggs import sum_product, FGG, Interpretation, FiniteFactor, json_to_fgg
from fggs.sum_product import scc, SumProduct
from fggs.semirings import *
import unittest, warnings, torch, random, json, copy, math

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
            Example('test/cycle.json', value=1-0.5**10, linear=True),
        ]

        for ex in self.examples:
            for fac in ex.fgg.interp.factors.values():
                if not isinstance(fac.weights, torch.Tensor):
                    fac.weights = torch.tensor(fac.weights, dtype=torch.get_default_dtype())


    def test_autograd(self):
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
                    opts = {'method': 'fixed-point', 'tol': 1e-6, 'kmax': 100, 'semiring': RealSemiring(dtype=torch.double)}
                    return SumProduct.apply(fgg, opts, in_labels, out_labels, *in_values)
                self.assertTrue(torch.autograd.gradcheck(f, in_values, atol=1e-3))

    def test_autograd_log(self):
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
                    opts = {'method': 'fixed-point', 'tol': 1e-6, 'kmax': 100, 'semiring': LogSemiring(dtype=torch.double)}
                    ret = SumProduct.apply(fgg, opts, in_labels, out_labels, *in_values)
                    # put exp inside f to avoid gradcheck computing -inf - -inf
                    return tuple(torch.exp(z) for z in ret)
                self.assertTrue(torch.autograd.gradcheck(f, in_values, atol=1e-3))

    def test_infinite_gradient(self):
        fgg = load_fgg('test/linear.json')
        # make sum-product infinite
        faca = fgg.interp.factors[fgg.grammar.get_edge_label('a')]
        faca.weights = torch.tensor(1., requires_grad=True)
        facb = fgg.interp.factors[fgg.grammar.get_edge_label('b')]
        facb.weights = torch.tensor(1., requires_grad=True)
        facc = fgg.interp.factors[fgg.grammar.get_edge_label('c')]
        facc.weights = torch.tensor(1., requires_grad=True)
        z = sum_product(fgg, method='linear')
        self.assertEqual(z.item(), math.inf)
        z.backward()
        self.assertEqual(faca.weights.grad.item(), math.inf)
        self.assertEqual(facb.weights.grad.item(), math.inf)
        self.assertEqual(facc.weights.grad.item(), math.inf)
        
        fgg = load_fgg('test/simplefgg.json')
        # make sum-product infinite
        fac1 = fgg.interp.factors[fgg.grammar.get_edge_label('fac1')]
        fac1.weights = torch.tensor(1., requires_grad=True)
        fac2 = fgg.interp.factors[fgg.grammar.get_edge_label('fac2')]
        fac2.weights = torch.tensor(1., requires_grad=True)
        z = sum_product(fgg, method='newton')
        self.assertEqual(z.item(), math.inf)
        z.backward()
        self.assertEqual(fac1.weights.grad.item(), math.inf)
        self.assertEqual(fac2.weights.grad.item(), math.inf)
        

    def test_sum_product(self):
        for method in ['fixed-point', 'linear', 'newton']:
            with self.subTest(method=method):
                for example in self.examples:
                    if method == 'linear' and not example.linear: continue
                    if method in ['linear', 'newton'] and not example.clean: continue # not implemented yet
                    with self.subTest(example=str(example)):
                        z_exact = example.exact()
                        fgg = example.fgg
                        with self.subTest(semiring='RealSemiring'):
                            z = sum_product(fgg, method=method, semiring=RealSemiring())
                            self.assertTrue(torch.norm(z - z_exact) < 1e-2,
                                            f'{z} != {z_exact}')
                        
                        interp = copy.deepcopy(fgg.interp)
                        for fac in interp.factors.values():
                            fac.weights = torch.log(fac.weights)
                        fgg = FGG(example.fgg.grammar, interp)
                        
                        with self.subTest(semiring='LogSemiring'):
                            z = torch.exp(sum_product(fgg, method=method, semiring=LogSemiring()))
                            self.assertTrue(torch.norm(z - z_exact) < 1e-2,
                                            f'{z} != {z_exact}')
                            
                        with self.subTest(semiring='ViterbiSemiring'):
                            z = torch.exp(sum_product(fgg, method=method, semiring=ViterbiSemiring()))

                            # Rerun at a very low temperature to estimate the correct value
                            
                            temp = 1/1000
                            for fac in interp.factors.values():
                                fac.weights /= temp
                            z_expected = torch.exp(temp * sum_product(fgg, method=method, semiring=LogSemiring()))
                            self.assertTrue(torch.norm(z - z_expected) < 1e-2,
                                            f'{z} != {z_expected}')

                        with self.subTest(semiring='BoolSemiring'):
                            interp = copy.deepcopy(example.fgg.interp)
                            for fac in interp.factors.values():
                                fac.weights = fac.weights > 0.
                            fgg = FGG(example.fgg.grammar, interp)
                            z = sum_product(fgg, method='fixed-point', semiring=BoolSemiring())
                            z_exact = example.exact() > 0.
                            self.assertTrue(torch.all(z == z_exact),
                                            f'{z} != {z_exact}')


if __name__ == '__main__':
    unittest.main()
