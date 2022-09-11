from fggs import sum_product, FGG, FiniteFactor, json_to_fgg
from fggs.sum_product import scc, SumProduct
from fggs.semirings import *
import unittest, warnings, torch, random, json, copy, math

def load_fgg(filename):
    with open(filename) as f:
        return json_to_fgg(json.load(f))

class Example:
    def __init__(self, filename, linear=False, gradcheck=True, value=None):
        self.name = filename
        self.linear = linear
        self.gradcheck = gradcheck
        if value is not None:
            self.value = value
        self.fgg = load_fgg(filename)

    def exact(self):
        return self.value

    def __str__(self):
        return self.name

class PPLExample(Example):
    def __init__(self, p):
        # Skip gradcheck because it's slow
        super().__init__('test/example12p.json', gradcheck=False)
        self.p = p
        self.fgg.factors['p'].weights = [1 - p, p]
        
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

class QuadraticExample(Example):
    def __init__(self, p):
        # Skip gradcheck near 0.5 because finite differences is not accurate
        gradcheck = abs(p-0.5) > 0.1
            
        super().__init__('test/simplefgg.json', gradcheck=gradcheck)
        self.p = p
        self.fgg.new_finite_factor('fac1', p)
        self.fgg.new_finite_factor('fac2', 1-p)
        
    def exact(self):
        p = self.p
        return min(1, (1-p)/p)

    def __str__(self):
        return f'{self.name} (p={self.p})'

class TestSumProduct(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings('ignore', message='.*maximum iteration.*')

        self.examples = [
            Example('test/hmm.json', value=1., linear=True, gradcheck=False), # gradcheck is slow
            PPLExample(0.1), PPLExample(0.5), PPLExample(0.9),
            QuadraticExample(0.25), QuadraticExample(0.5), QuadraticExample(0.75), QuadraticExample(1.),
            Example('test/disconnected_node.json', value=torch.tensor([18., 18., 18.]), linear=True),
            Example('test/barhillel.json', value=torch.tensor([[0.1129, 0.0129], [0.0129, 0.1129]])),
            Example('test/test.json', value=torch.tensor([7., 1., 1.]), gradcheck=False), # gradcheck tries negative weights, which we don't support
            Example('test/linear.json', value=7.5, linear=True),
            Example('test/cycle.json', value=1-0.5**10, linear=True),
        ]

        for ex in self.examples:
            for fac in ex.fgg.factors.values():
                if not isinstance(fac.weights, torch.Tensor):
                    fac.weights = torch.tensor(fac.weights, dtype=torch.get_default_dtype())


    def assertAlmostEqual(self, x, y):
        if x.dtype is torch.bool:
            self.assertTrue(torch.all(x == y), f'{x} != {y}')
        else:
            x = torch.as_tensor(x, dtype=torch.get_default_dtype())
            y = torch.as_tensor(y, dtype=torch.get_default_dtype())
            self.assertTrue(torch.allclose(x.nan_to_num(), y.nan_to_num(), rtol=1e-2, atol=1e-2), f'{x} != {y}')
            self.assertFalse(torch.any(x.isnan()))
            self.assertFalse(torch.any(y.isnan()))
            self.assertTrue(torch.all(x.isinf() == y.isinf()), f'{x} != {y}')
            
    def test_autograd(self):
        for example in self.examples:
            if not example.gradcheck: continue
            with self.subTest(example=str(example)):
                fgg = example.fgg
                in_labels = fgg.terminals()
                in_values = [fac.weights.to(torch.double).requires_grad_(True)
                             for fac in fgg.factors.values()]
                out_labels = list(fgg.nonterminals())
                def f(*in_values):
                    opts = {'method': 'newton', 'tol': 1e-6, 'kmax': 100,
                            'semiring': RealSemiring(dtype=torch.double)}
                    return SumProduct.apply(fgg, opts, in_labels, out_labels, *in_values)
                
                self.assertTrue(torch.autograd.gradcheck(f, in_values, atol=1e-3))


    def test_autograd_log(self):
        for example in self.examples:
            if not example.gradcheck: continue
            with self.subTest(example=str(example)):
                fgg = example.fgg
                in_labels = fgg.terminals()
                in_values = [fac.weights.log().to(torch.double).requires_grad_(True)
                             for fac in fgg.factors.values()]
                out_labels = list(fgg.nonterminals())
                def f(*in_values):
                    opts = {'method': 'newton', 'tol': 1e-6, 'kmax': 100,
                            'semiring': LogSemiring(dtype=torch.double)}
                    ret = SumProduct.apply(fgg, opts, in_labels, out_labels, *in_values)
                    # put exp inside f to avoid gradcheck computing -inf - -inf
                    return tuple(torch.exp(z) for z in ret)
                self.assertTrue(torch.autograd.gradcheck(f, in_values, atol=1e-3))

    def test_infinite_gradient(self):
        fgg = load_fgg('test/linear.json')
        # make sum-product infinite
        faca = fgg.factors['a']
        faca.weights = torch.tensor(1., requires_grad=True)
        facb = fgg.factors['b']
        facb.weights = torch.tensor(1., requires_grad=True)
        facc = fgg.factors['c']
        facc.weights = torch.tensor(1., requires_grad=True)
        z = sum_product(fgg, method='linear')
        self.assertEqual(z.item(), math.inf)
        z.backward()
        self.assertEqual(faca.weights.grad.item(), math.inf)
        self.assertEqual(facb.weights.grad.item(), math.inf)
        self.assertEqual(facc.weights.grad.item(), math.inf)
        
        fgg = load_fgg('test/simplefgg.json')
        # make sum-product infinite
        fgg.new_finite_factor('fac1', torch.tensor(1., requires_grad=True))
        fgg.new_finite_factor('fac2', torch.tensor(1., requires_grad=True))
        z = sum_product(fgg, method='newton')
        self.assertEqual(z.item(), math.inf)
        z.backward()
        self.assertEqual(fgg.factors['fac1'].weights.grad.item(), math.inf)
        self.assertEqual(fgg.factors['fac2'].weights.grad.item(), math.inf)
        

    def test_sum_product(self):
        for method in ['fixed-point', 'linear', 'newton']:
            with self.subTest(method=method):
                for example in self.examples:
                    if method == 'linear' and not example.linear: continue
                    with self.subTest(example=str(example)):
                        z_exact = example.exact()
                        with self.subTest(semiring='RealSemiring'):
                            z = sum_product(example.fgg, method=method, semiring=RealSemiring())
                            self.assertAlmostEqual(z, z_exact)

                        with self.subTest(semiring='LogSemiring'):
                            # Take log of all weights
                            fgg = example.fgg.copy()
                            for fac in fgg.factors.values():
                                fac.weights = torch.log(fac.weights)
                            z = torch.exp(sum_product(fgg, method=method, semiring=LogSemiring()))
                            self.assertAlmostEqual(z, z_exact)
                            
                        with self.subTest(semiring='ViterbiSemiring'):
                            # Reuse weights from LogSemiring
                            z = torch.exp(sum_product(fgg, method=method, semiring=ViterbiSemiring()))
                            # Rerun at a very low temperature to estimate the correct value
                            temp = 1/1000
                            for fac in fgg.factors.values():
                                fac.weights /= temp
                            z_expected = torch.exp(temp * sum_product(fgg, method=method, semiring=LogSemiring()))
                            self.assertAlmostEqual(z, z_expected)

                        with self.subTest(semiring='BoolSemiring'):
                            fgg = example.fgg.copy()
                            for fac in fgg.factors.values():
                                fac.weights = fac.weights > 0.
                            z = sum_product(fgg, method=method, semiring=BoolSemiring())
                            z_exact = example.exact() > 0.
                            self.assertAlmostEqual(z, z_exact)


if __name__ == '__main__':
    unittest.main()
