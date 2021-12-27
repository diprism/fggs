import unittest
from fggs import *

class TestFiniteDomain(unittest.TestCase):
    
    def setUp(self):
        self.dom    = FiniteDomain(['foo', 'bar', 'baz'])
        self.dom_eq = FiniteDomain(['foo', 'bar', 'baz'])
        self.dom_ne = FiniteDomain(['qux', 'quux', 'quuz'])
    
    def test_basic(self):
        self.assertTrue(self.dom.contains('foo'))
        self.assertFalse(self.dom.contains('qux'))
        self.assertEqual(self.dom.values, ['foo', 'bar', 'baz'])
        self.assertEqual(self.dom.size(), 3)
        self.assertEqual(self.dom.numberize('bar'), 1)
        self.assertEqual(self.dom.denumberize(1), 'bar')
    
    def test_equals(self):
        self.assertFalse(self.dom == None)
        self.assertFalse(self.dom == 5)
        self.assertTrue(self.dom == self.dom_eq)
        self.assertFalse(self.dom == self.dom_ne)
        self.assertFalse(self.dom != self.dom_eq)
        self.assertTrue(self.dom != self.dom_ne)

    def test_hash(self):
        with self.assertRaises(TypeError):
            d = dict()
            d[self.dom] = 5
        
if __name__ == "__main__":
    unittest.main()
    
