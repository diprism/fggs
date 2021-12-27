import unittest
import json
import os
import copy
import fggs

class TestJSON(unittest.TestCase):
    def test_erase_ids(self):
        
        # When an HRG is created without explicit node/edge ids,
        # unique ids are automatically chosen. We can read a JSON file
        # without explicit ids and write it back to JSON, but checking
        # that the result is equal would require checking for graph
        # isomorphism. So we don't bother, but we do check that the
        # written JSON also doesn't have explicit ids.
        
        for filename in ['test.json']:
            with open(os.path.join(os.path.dirname(__file__), filename)) as f:
                j = json.load(f)['grammar']
            g = fggs.json_to_hrg(j)
            j_check = fggs.hrg_to_json(g)

            for r in j_check['rules']:
                for n in r['rhs']['nodes']:
                    self.assertTrue('id' not in n)
    
    def test_roundtrip(self):
        for filename in ['hmm.json', 'example12p.json', 'simplefgg.json']:
            with open(os.path.join(os.path.dirname(__file__), filename)) as f:
                j = json.load(f)
            fgg = fggs.json_to_fgg(j)
            j_check = fggs.fgg_to_json(fgg)

            jg = j['grammar']
            jg_check = j_check['grammar']

            self.maxDiff = 10000
            self.assertEqual(jg.keys(), jg_check.keys())
            self.assertEqual(jg['terminals'], jg_check['terminals'])
            self.assertEqual(jg['nonterminals'], jg_check['nonterminals'])
            self.assertEqual(jg['start'], jg_check['start'])

            # ignore order of rules
            for r in jg['rules']:
                self.assertTrue(r in jg_check['rules'], f'rule: {r}\nnot found in: {jg_check["rules"]}')
            for r in jg_check['rules']:
                self.assertTrue(r in jg['rules'], r)

            ji = j['interpretation']
            ji_check = j['interpretation']
            self.assertEqual(ji['domains'], ji_check['domains'])
            self.assertEqual(ji['factors'], ji_check['factors'])

    def test_error(self):
        with open(os.path.join(os.path.dirname(__file__), 'hmm.json')) as f:
            j = json.load(f)
            jcopy = copy.deepcopy(j)
            jcopy['grammar']['rules'][0]['rhs']['edges'][0]['attachments'] = [100]
            with self.assertRaises(ValueError):
                _ = fggs.json_to_fgg(jcopy)
            jcopy = copy.deepcopy(j)
            jcopy['grammar']['rules'][0]['rhs']['externals'] = [100]
            with self.assertRaises(ValueError):
                _ = fggs.json_to_fgg(jcopy)

if __name__ == "__main__":
    unittest.main()
    
