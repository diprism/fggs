import unittest

import torch
import fggs
from fggs.adjoint import adjoint_hrg

class TestAdjoint(unittest.TestCase):
    def test_adjoint(self):
        g = fggs.HRG()

        s = fggs.EdgeLabel("S", (), is_nonterminal=True)
        g.start_symbol = s

        p = 1/3
        pfac = fggs.EdgeLabel("p", (), is_terminal=True)

        rhs = fggs.Graph()
        rhs.add_edge(fggs.Edge(s, ()))
        rhs.add_edge(fggs.Edge(pfac, ()))
        g.add_rule(fggs.HRGRule(s, rhs))

        rhs = fggs.Graph()
        g.add_rule(fggs.HRGRule(s, rhs))

        gi = fggs.Interpretation()
        top = fggs.EdgeLabel("top", (), is_terminal=True)
        gi.add_factor(top, fggs.CategoricalFactor([], torch.tensor(1)))
        gi.add_factor(pfac, fggs.CategoricalFactor([], torch.tensor(p)))

        #z = fggs.sum_product(fggs.FGG(g, gi)) # 1/(1-p)

        gbar, index = adjoint_hrg(g, {s: top})
        gbar.start_symbol = index[pfac]
        z = fggs.sum_product(fggs.FGG(gbar, gi)) # 1/(1-p)²

        self.assertAlmostEqual(z.item(), 1/(1-p)**2, places=5)



