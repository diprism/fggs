from sum_product import sum_product
from formats import json_to_fgg
import unittest, warnings, random, json

class TestSumProduct(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings('ignore', message='.*maximum iteration.*')
        file = open('test/hmm.json')
        self.fgg_1 = json_to_fgg(json.load(file))
        file.close()
        file = open('test/example12p.json')
        self.fgg_2 = json_to_fgg(json.load(file))
        file.close()

    def test_fixed_point_1(self):
        self.assertAlmostEqual(sum_product(self.fgg_1, method='fixed-point').item(), 1.0)

    def test_fixed_point_2(self):
        from math import sqrt
        def exact_value(p):
            # minimal solution of (x, y) = (2pxy + (1 - p), p(x^2 + y^2)) where x = p(true) and y = p(false)
            return ((3 - 2*p - sqrt(1 + 4*p - 4*p**2))/(4*p), ( 1 - 2*p + sqrt(1 + 4*p - 4*p**2))/(4*p)) if p > 0.5 \
              else ((1 + 2*p - sqrt(1 + 4*p - 4*p**2))/(4*p), (-1 + 2*p + sqrt(1 + 4*p - 4*p**2))/(4*p))
        for p in (random.uniform(0.01, 0.99) for _ in range(50)):
            self.fgg_2.get_terminal('p').factor()._weights = [1 - p, p]
            for A, B in zip(sum_product(self.fgg_2, method='fixed-point'), exact_value(p)):
                self.assertAlmostEqual(A.item(), B, places=2)

    def test_broyden_1(self):
        self.assertAlmostEqual(sum_product(self.fgg_1, method='broyden').item(), 1.0, places=6)

    def test_broyden_2(self):
        from math import sqrt
        def exact_value(p):
            # minimal solution of (x, y) = (2pxy + (1 - p), p(x^2 + y^2)) where x = p(true) and y = p(false)
            return ((3 - 2*p - sqrt(1 + 4*p - 4*p**2))/(4*p), ( 1 - 2*p + sqrt(1 + 4*p - 4*p**2))/(4*p)) if p > 0.5 \
              else ((1 + 2*p - sqrt(1 + 4*p - 4*p**2))/(4*p), (-1 + 2*p + sqrt(1 + 4*p - 4*p**2))/(4*p))
        for p in (random.uniform(0.01, 0.99) for _ in range(50)):
            self.fgg_2.get_terminal('p').factor()._weights = [1 - p, p]
            try:
                for A, B in zip(sum_product(self.fgg_2, method='broyden'), exact_value(p)):
                    self.assertAlmostEqual(A.item(), B, places=2)
            except AssertionError as e:
                k = 0
                while k < 50:
                    perturbation = round(random.uniform(0.01, 0.99), 1)
                    try:
                        for A, B in zip(sum_product(self.fgg_2, method='broyden', perturbation=perturbation), exact_value(p)):
                            self.assertAlmostEqual(A.item(), B, places=2)
                        break
                    except AssertionError:
                        pass
                    k += 1
                else: raise e

if __name__ == '__main__':
    unittest.main()
