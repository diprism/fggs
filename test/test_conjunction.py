import unittest
import json
import os
import fggs
from fggs.conjunction import check_namespace_collisions, nonterminal_pairs, conjoinable, conjoin_rules, conjoin_hrgs


class TestConjunction(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(__file__), 'hmm.json')) as f:
            self.hmm_json = json.load(f)['grammar']
        with open(os.path.join(os.path.dirname(__file__), 'conjunct.json')) as f:
            self.conjunct_json = json.load(f)
        with open(os.path.join(os.path.dirname(__file__), 'conjunction.json')) as f:
            self.conjunction_json = json.load(f)        
        self.restore()

    def restore(self):
        self.hmm = fggs.json_to_hrg(self.hmm_json)
        self.conjunct = fggs.json_to_hrg(self.conjunct_json)
        self.conjunction = fggs.json_to_hrg(self.conjunction_json)
        
        # extract specific rules for testing conjunction
        xrules1 = self.hmm.rules(self.hmm.get_edge_label("X"))
        for rule in xrules1:
            if len(rule.rhs.nodes()) == 3:
                self.xrule1 = rule
        
        xrules2 = self.conjunct.rules(self.conjunct.get_edge_label("X"))
        for rule in xrules2:
            if "Y" in [nt.name for nt in rule.rhs.nonterminals()]:
                self.xrule2 = rule
        
        xrules3 = self.conjunction.rules(self.conjunction.get_edge_label("<X,X>"))
        for rule in xrules3:
            if "Y" in [nt.name for nt in rule.rhs.nonterminals()]:
                self.xrule3 = rule
        
        # get all the nodes for easy access
        self.nodes1 = {node.id:node for node in self.xrule1.rhs.nodes()}
        self.nodes2 = {node.id:node for node in self.xrule2.rhs.nodes()}

        # extract node labels and edge labels for use in testing
        self.nl_t = self.hmm.get_node_label("T")
        self.nl_w = self.hmm.get_node_label("W")
        self.el_x1 = self.hmm.get_edge_label("X")
        self.el_x2 = self.conjunct.get_edge_label("X")

    def test_check_namespace_collisions(self):
        nl = fggs.NodeLabel("nl")
        el1 = fggs.EdgeLabel("collide_el", [nl], is_terminal=True)
        el2 = fggs.EdgeLabel("collide_el", [nl,nl], is_terminal=True)
        el3 = fggs.EdgeLabel("don't_collide", [nl], is_nonterminal=True)
        self.hmm.add_node_label(nl)
        self.conjunct.add_node_label(nl)
        self.hmm.add_edge_label(el1)
        self.conjunct.add_edge_label(el2)
        self.conjunct.add_edge_label(el3)
        (n, e) = check_namespace_collisions(self.hmm, self.conjunct)
        self.assertEqual(len(n), 0)
        self.assertEqual(len(e), 1)
        with self.assertRaises(ValueError):
            conjoin_hrgs(self.hmm, self.conjunct)

    def test_nonterminal_pairs(self):
        s = fggs.EdgeLabel("S", [], is_nonterminal=True)
        hrg1 = fggs.HRG(s)
        hrg2 = fggs.HRG(s)
        hrg1.add_edge_label(fggs.EdgeLabel(name="X", is_nonterminal=True, node_labels=()))
        hrg2.add_edge_label(fggs.EdgeLabel(name="Y,Z", is_nonterminal=True, node_labels=()))
        hrg1.add_edge_label(fggs.EdgeLabel(name="X,Y", is_nonterminal=True, node_labels=()))
        hrg2.add_edge_label(fggs.EdgeLabel(name="Z", is_nonterminal=True, node_labels=()))
        hrg1.add_edge_label(fggs.EdgeLabel(name="<X,Z>", is_terminal=True, node_labels=()))
        nt_map = nonterminal_pairs(hrg1, hrg2)
        self.assertEqual(sorted(nt.name for nt in nt_map.values()), ["<S,S>", "<S,Y,Z>", "<S,Z>", "<X,S>", "<X,Y,S>", "<X,Y,Y,Z>", "<X,Y,Z>", "<X,Y,Z>_1", "<X,Z>_1"])

    def test_conjoinable(self):    
        self.assertTrue(conjoinable(self.xrule1, self.xrule2))        
        # rules have different numbers of nodes
        self.xrule1.rhs.add_node(fggs.Node(label=self.nl_t,id="v3"))
        self.assertFalse(conjoinable(self.xrule1, self.xrule2))
        self.restore()
        # rules have same number of nodes, different ids
        self.xrule1.rhs.add_node(fggs.Node(label=self.nl_t,id="v3"))
        self.xrule2.rhs.add_node(fggs.Node(label=self.nl_t,id="v4"))
        self.assertFalse(conjoinable(self.xrule1, self.xrule2))
        self.restore()
        # rules have same number of nodes, same ids, different labels
        self.xrule1.rhs.add_node(fggs.Node(label=self.nl_t,id="v3"))
        self.xrule2.rhs.add_node(fggs.Node(label=self.nl_w,id="v3"))
        self.assertFalse(conjoinable(self.xrule1, self.xrule2))
        self.restore()
        # rules have different numbers of nonterminals
        self.xrule1.rhs.add_edge(fggs.Edge(label=self.el_x1,nodes=[self.nodes1["v0"]],id="e4"))
        self.assertFalse(conjoinable(self.xrule1, self.xrule2))
        self.restore()
        # rules have same number of nonterminals, different ids
        self.xrule1.rhs.add_edge(fggs.Edge(label=self.el_x1,nodes=[self.nodes1["v0"]],id="e4"))
        self.xrule2.rhs.add_edge(fggs.Edge(label=self.el_x2,nodes=[self.nodes2["v0"]],id="e5"))
        self.assertFalse(conjoinable(self.xrule1, self.xrule2))
        self.restore()
        # rules have same number of nonterminals, same ids, different attachments
        self.xrule1.rhs.add_edge(fggs.Edge(label=self.el_x1,nodes=[self.nodes1["v0"]],id="e4"))
        self.xrule2.rhs.add_edge(fggs.Edge(label=self.el_x2,nodes=[self.nodes2["v1"]],id="e4"))
        self.assertFalse(conjoinable(self.xrule1, self.xrule2))
        self.restore()
        # rules have different external nodes
        self.xrule1.rhs.ext = [self.nodes1["v1"]]
        self.assertFalse(conjoinable(self.xrule1, self.xrule2))
        self.restore()

    def test_conjunction(self):
        conjunction_check = conjoin_hrgs(self.hmm, self.conjunct)
        conjunction_json_check = fggs.hrg_to_json(conjunction_check)
        self.maxDiff = 10000
        self.assertEqual(self.conjunction_json.keys(), conjunction_json_check.keys())
        self.assertEqual(self.conjunction_json['terminals'], conjunction_json_check['terminals'])
        self.assertEqual(self.conjunction_json['nonterminals'], conjunction_json_check['nonterminals'])
        self.assertEqual(self.conjunction_json['start'], conjunction_json_check['start'])

        # ignore order of rules
        for r in self.conjunction_json['rules']:
            self.assertTrue(r in conjunction_json_check['rules'])
        for r in conjunction_json_check['rules']:
            self.assertTrue(r in self.conjunction_json['rules'])
