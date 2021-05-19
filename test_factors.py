import unittest
from domains import *
from factors import *
import torch

class TestCategoricalFactor(unittest.TestCase):
    def test_basic(self):
        d3 = FiniteDomain('d3', ['foo', 'bar', 'baz'])
        d4 = FiniteDomain('d4', ['jia', 'yi', 'bing', 'ding'])
        f = CategoricalFactor('f', [d3, d4], torch.arange(12).reshape(3, 4))
        self.assertEqual(f.apply(['foo', 'jia']), 0)
        self.assertEqual(f.apply(['foo', 'yi']), 1)
        self.assertEqual(f.apply(['bar', 'jia']), 4)
        self.assertEqual(f.apply(['baz', 'ding']), 11)

if __name__ == "__main__":
    unittest.main()
    
