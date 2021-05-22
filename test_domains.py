import unittest
from domains import *

class TestFiniteDomain(unittest.TestCase):
    def test_basic(self):
        dom = FiniteDomain(['foo', 'bar', 'baz'])
        self.assertTrue(dom.contains('foo'))
        self.assertFalse(dom.contains('qux'))
        self.assertEqual(dom.values(), ['foo', 'bar', 'baz'])
        self.assertEqual(dom.size(), 3)
        self.assertEqual(dom.numberize('bar'), 1)
        
if __name__ == "__main__":
    unittest.main()
    
