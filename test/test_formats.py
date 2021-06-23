import unittest
import fgg_representation as fggs
import formats
import json
import os
import copy

class TestJson(unittest.TestCase):

    def tearDown(self):
        fggs.NodeLabel.clear_registry()
        fggs.EdgeLabel.clear_registry()

    def test_roundtrip(self):
        for filename in ['hmm.json', 'example12p.json']:
            with open(os.path.join(os.path.dirname(__file__), filename)) as f:
                j = json.load(f)
            g = formats.json_to_fgg(j)
            j_check = formats.fgg_to_json(g)

            self.maxDiff = 10000
            self.assertEqual(j.keys(), j_check.keys())
            self.assertEqual(j['domains'], j_check['domains'])
            self.assertEqual(j['factors'], j_check['factors'])
            self.assertEqual(j['nonterminals'], j_check['nonterminals'])
            self.assertEqual(j['start'], j_check['start'])

            # ignore order of rules
            for r in j['rules']:
                self.assertTrue(r in j_check['rules'], f'rule: {r}\nnot found in: {j_check["rules"]}')
            for r in j_check['rules']:
                self.assertTrue(r in j['rules'], r)

    def test_error(self):
        with open(os.path.join(os.path.dirname(__file__), 'hmm.json')) as f:
            j = json.load(f)
            jcopy = copy.deepcopy(j)
            jcopy['rules'][0]['rhs']['edges'][0]['attachments'] = [100]
            with self.assertRaises(ValueError):
                _ = formats.json_to_fgg(jcopy)
            jcopy = copy.deepcopy(j)
            jcopy['rules'][0]['rhs']['externals'] = [100]
            with self.assertRaises(ValueError):
                _ = formats.json_to_fgg(jcopy)

if __name__ == "__main__":
    unittest.main()
    
