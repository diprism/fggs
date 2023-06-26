import torch
import fggs
import unittest
import json
import sys

class TestReadme(unittest.TestCase):

    def test_readme(self):

        fgg = fggs.FGG('S')

        rhs = fggs.Graph()

        v1 = rhs.new_node('T')

        rhs.new_edge('is_bos', [v1], is_terminal=True)
        rhs.new_edge('X', [v1], is_nonterminal=True)

        fgg.new_rule('S', rhs)

        rhs = fggs.Graph()
        v1, v2, v3 = rhs.new_node('T'), rhs.new_node('T'), rhs.new_node('W')
        rhs.new_edge('transition', [v1, v2], is_terminal=True)
        rhs.new_edge('observation', [v2, v3], is_terminal=True)
        rhs.new_edge('X', [v2], is_nonterminal=True)
        rhs.ext = [v1]
        fgg.new_rule('X', rhs)

        rhs = fggs.Graph()
        v1, v2 = rhs.new_node('T'), rhs.new_node('T')
        rhs.new_edge('transition', [v1, v2], is_terminal=True)
        rhs.new_edge('is_eos', [v2], is_terminal=True)
        rhs.ext = [v1]
        fgg.new_rule('X', rhs)

        fgg.new_finite_domain('T', ['BOS', 'EOS', 'IN', 'NNS', 'VBP'])
        fgg.new_finite_domain('W', ['cats', 'chase', 'dogs', 'that'])

        fgg.new_finite_factor('is_bos', torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))
        fgg.new_finite_factor('is_eos', torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]))
        fgg.new_finite_factor('transition',
            torch.tensor([
                [0.0, 0.0, 0.0, 1.0, 0.0], # BOS
                [0.0, 0.0, 0.0, 0.0, 0.0], # EOS
                [0.0, 0.0, 0.0, 0.0, 1.0], # IN
                [0.0, 0.5, 0.5, 0.0, 0.0], # NNS
                [0.0, 0.0, 0.0, 1.0, 0.0], # VBP
            ])
        )
        fgg.new_finite_factor('observation',
            torch.tensor([
                [0.0, 0.0, 0.0, 0.0], # BOS
                [0.0, 0.0, 0.0, 0.0], # EOS
                [0.0, 0.0, 0.0, 1.0], # IN
                [0.5, 0.0, 0.5, 0.0], # NNS
                [0.0, 1.0, 0.0, 0.0], # VBP
            ])
        )

        j = fggs.fgg_to_json(fgg)
        self.assertEqual(j, json.loads(json.dumps(j)))

        self.assertAlmostEqual(fggs.sum_product(fgg, tol=1e-10).item(), 1)

if __name__ == '__main__':
    unittest.main()
