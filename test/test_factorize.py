import unittest
from fggs import *
from fggs.factorize import add_node, add_edge, tree_decomposition
from fggs.utils import naive_graph_isomorphism
import re, collections, os, json

class TestTreewidth(unittest.TestCase):
    def check(self, filename, optimal_tw, method, exact):
        g = {}
        with open(filename) as f:
            for line in f:
                if line[0] not in 'cp':
                    u, v = line.split()
                    add_node(g, u)
                    add_node(g, v)
                    add_edge(g, u, v)
        t = tree_decomposition(g, method=method)

        # vertex cover
        for v in g:
            found = False
            for b in t:
                if v in b:
                    found = True
            self.assertTrue(found)

        # edge cover
        for u in g:
            for v in g[u]:
                found = False
                for b in t:
                    if u in b and v in b:
                        found = True
                self.assertTrue(found)

        # running intersection
        count = collections.Counter()
        def visit(bag, parent):
            for v in bag:
                if parent is None or v not in parent:
                    count[v] += 1
            for n in t[bag]:
                if n != parent:
                    visit(n, bag)
        visit(list(t)[0], None)
        for v in count:
            self.assertEqual(count[v], 1, f'node {v} violates running intersection')

        tw = max(len(b) for b in t)-1
        if exact:
            self.assertEqual(tw, optimal_tw)
        else:
            self.assertTrue(tw >= optimal_tw)

    # Test cases from https://github.com/freetdi/named-graphs/
    def test_min_fill(self):
        dirname = os.path.dirname(__file__)
        for filename, optimal_tw in [
                ('BarbellGraph_10_5.gr', 9),
                ('BidiakisCube.gr', 4),
                ('BlanusaFirstSnarkGraph.gr', 5),
                ('BlanusaSecondSnarkGraph.gr', 4),
                ('BrinkmannGraph.gr', 8),
        ]:
            self.check(os.path.join(dirname, 'graphs', filename), optimal_tw, 'min_fill', False)
            
    def test_acb(self):
        dirname = os.path.dirname(__file__)
        for filename, optimal_tw in [
                #('BarbellGraph_10_5.gr', 9),
                ('BidiakisCube.gr', 4),
                ('BlanusaFirstSnarkGraph.gr', 5),
                ('BlanusaSecondSnarkGraph.gr', 4),
                #('BrinkmannGraph.gr', 8),
        ]:
            self.check(os.path.join(dirname, 'graphs', filename), optimal_tw, 'acb', True)
            
    def test_quickbb(self):
        dirname = os.path.dirname(__file__)
        for filename, optimal_tw in [
                ('BarbellGraph_10_5.gr', 9),
                ('BidiakisCube.gr', 4),
                #('BlanusaFirstSnarkGraph.gr', 5),
                #('BlanusaSecondSnarkGraph.gr', 4),
                #('BrinkmannGraph.gr', 8),
        ]:
            self.check(os.path.join(dirname, 'graphs', filename), optimal_tw, 'quickbb', True)

class TestFactorize(unittest.TestCase):
    def test_factorize(self):
        for filename in ['hmm.json', 'simplefgg.json']:
            with self.subTest(filename):
                with open(os.path.join(os.path.dirname(__file__), filename)) as f:
                    j = json.load(f)['grammar']
                g = json_to_hrg(j)
                for r in g.all_rules():
                    frules = {fr.lhs:fr for fr in factorize_rule(r)}
                    def visit(fr):
                        for e in list(fr.rhs.edges()):
                            if e.label.is_nonterminal and e.label not in g.nonterminals():
                                visit(frules[e.label])
                                replace_edge(fr.rhs, e, frules[e.label].rhs)
                    rr = frules[r.lhs]
                    visit(rr)
                    self.assertEqual(r.lhs, rr.lhs)
                    result, m = naive_graph_isomorphism(r.rhs, rr.rhs)
                    self.assertTrue(result, m)
            
if __name__ == "__main__":
    unittest.main()
