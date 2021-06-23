from sum_product import sum_product
from formats import json_to_fgg
import unittest, random, json

class TestSumProduct(unittest.TestCase):

    def setUp(self):
        file = open('test/hmm.json')
        self.fgg_1 = json_to_fgg(json.load(file))
        file.close()
        file = open('test/example12p.json')
        self.fgg_2 = json_to_fgg(json.load(file))
        file.close()

    def test_fixed_point_1(self):
        self.assertAlmostEqual(sum_product(self.fgg_1, method='fixed-point'), 1.0)

    def test_fixed_point_2(self):
        for p in (random.uniform(0.51, 0.99) for _ in range(100)):
            self.fgg_2.get_terminal('p').factor()._weights = [1 - p, p]
            self.assertAlmostEqual(sum_product(self.fgg_2, method='fixed-point'), (1 - p)/p, places=2)

if __name__ == '__main__':
    unittest.main()
