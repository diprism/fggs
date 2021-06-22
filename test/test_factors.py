import unittest
from domains import *
from factors import *

class TestCategoricalFactor(unittest.TestCase):

    def setUp(self):
        self.d3 = FiniteDomain(['foo', 'bar', 'baz'])
        self.d4 = FiniteDomain(['jia', 'yi', 'bing', 'ding'])

    def test_basic(self):
        f = CategoricalFactor([self.d3, self.d4], [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ])
        self.assertEqual(f.apply(['foo', 'jia']), 0)
        self.assertTrue(isinstance(f.apply(['foo', 'jia']), float))
        self.assertEqual(f.apply(['foo', 'yi']), 1)
        self.assertEqual(f.apply(['bar', 'jia']), 4)
        self.assertEqual(f.apply(['baz', 'ding']), 11)

    def test_failure(self):
        with self.assertRaises(ValueError):
            f = CategoricalFactor([self.d3, self.d4], [0, 1, 2, 3])
        with self.assertRaises(ValueError):
            f = CategoricalFactor([self.d3, self.d4], [[[0]]])
        with self.assertRaises(ValueError):
            f = CategoricalFactor([self.d3, self.d4], [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10],
            ])
        with self.assertRaises(ValueError):
            f = CategoricalFactor([self.d3, self.d4], [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [8, 9, 10, 11],
            ])

    def test_equals(self):
        f = CategoricalFactor([self.d3, self.d4], [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ])
        f_eq = CategoricalFactor([self.d3, self.d4], [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ])
        f_ne_doms = CategoricalFactor([self.d3, self.d3], [
            [0, 1, 2],
            [4, 5, 6],
            [8, 9, 10],
        ])
        f_ne_vals = CategoricalFactor([self.d3, self.d4], [
            [12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23],
        ])
        self.assertTrue(f == f_eq)
        self.assertFalse(f == f_ne_doms)
        self.assertFalse(f == f_ne_vals)
        self.assertFalse(f != f_eq)
        self.assertTrue(f != f_ne_doms)
        self.assertTrue(f != f_ne_vals)
    
    def test_hash(self):
        f = CategoricalFactor([self.d3, self.d4], [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ])
        with self.assertRaises(TypeError):
            d = dict()
            d[f] = 5

if __name__ == "__main__":
    unittest.main()
