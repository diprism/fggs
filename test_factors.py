import unittest
from domains import *
from factors import *

class TestCategoricalFactor(unittest.TestCase):
    def test_basic(self):
        d3 = FiniteDomain(['foo', 'bar', 'baz'])
        d4 = FiniteDomain(['jia', 'yi', 'bing', 'ding'])
        f = CategoricalFactor([d3, d4], [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ])
        self.assertEqual(f.apply(['foo', 'jia']), 0)
        self.assertEqual(f.apply(['foo', 'yi']), 1)
        self.assertEqual(f.apply(['bar', 'jia']), 4)
        self.assertEqual(f.apply(['baz', 'ding']), 11)

if __name__ == "__main__":
    unittest.main()
    
