import unittest
import formats
import json
import os

class TestJson(unittest.TestCase):
    def test_roundtrip(self):
        with open(os.path.join(os.path.dirname(__file__), 'hmm.json')) as f:
            j = json.load(f)
        g = formats.json_to_fgg(j)
        j_check = formats.fgg_to_json(g)
        self.maxDiff = 10000
        self.assertEqual(j, j_check)

if __name__ == "__main__":
    unittest.main()
    
