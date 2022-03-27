import unittest
from fggs import *

class TestFiniteFactor(unittest.TestCase):

    def setUp(self):
        self.d3 = FiniteDomain(['foo', 'bar', 'baz'])
        self.d4 = FiniteDomain(['jia', 'yi', 'bing', 'ding'])
        self.f = FiniteFactor([self.d3, self.d4], [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ])

    def test_basic(self):
        self.assertEqual(self.f.apply(['foo', 'jia']), 0)
        self.assertEqual(self.f.apply(['foo', 'yi']), 1)
        self.assertEqual(self.f.apply(['bar', 'jia']), 4)
        self.assertEqual(self.f.apply(['baz', 'ding']), 11)

    def test_failure(self):
        with self.assertRaises(ValueError):
            f = FiniteFactor([self.d3, self.d4], [0, 1, 2, 3])
        with self.assertRaises(ValueError):
            f = FiniteFactor([self.d3, self.d4], [[[0]]])
        with self.assertRaises(ValueError):
            f = FiniteFactor([self.d3, self.d4], [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10],
            ])
        with self.assertRaises(ValueError):
            f = FiniteFactor([self.d3, self.d4], [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [8, 9, 10, 11],
            ])

    def test_equals(self):
        f_eq = FiniteFactor([self.d3, self.d4], [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ])
        f_ne_doms = FiniteFactor([self.d3, self.d3], [
            [0, 1, 2],
            [4, 5, 6],
            [8, 9, 10],
        ])
        f_ne_vals = FiniteFactor([self.d3, self.d4], [
            [12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23],
        ])
        self.assertFalse(self.f == None)
        self.assertFalse(self.f == 5)
        self.assertTrue(self.f == f_eq)
        self.assertFalse(self.f == f_ne_doms)
        self.assertFalse(self.f == f_ne_vals)
        self.assertFalse(self.f != f_eq)
        self.assertTrue(self.f != f_ne_doms)
        self.assertTrue(self.f != f_ne_vals)

    def test_hash(self):
        with self.assertRaises(TypeError):
            d = dict()
            d[self.f] = 5

if __name__ == "__main__":
    unittest.main()
    
