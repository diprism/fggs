import unittest
import json
import os
import formats
import fgg_representation as fggs
from conjunction import conjoinable, conjoin_rules, conjoin_fggs


class TestConjunction(unittest.TestCase):

    def restore(self):
        self.hmm = formats.json_to_fgg(self.hmm_json)
        self.conjoinand = formats.json_to_fgg(self.conjoinand_json)
        self.conjunction = formats.json_to_fgg(self.conjunction_json)
        
        # extract specific rules for testing conjunction
        xrules1 = self.hmm.rules("X")
        for rule in xrules1:
            if len(rule.rhs().nodes()) == 3:
                self.xrule1 = rule
        
        xrules2 = self.conjoinand.rules("X")
        for rule in xrules2:
            if "Y" in [nt.label().name() for nt in rule.rhs().nonterminals()]:
                self.xrule2 = rule
        
        xrules3 = self.conjunction.rules("<X,X>")
        for rule in xrules3:
            if "Y" in [nt.label().name() for nt in rule.rhs().nonterminals()]:
                self.xrule3 = rule
        
        # get all the nodes for easy access
        self.nodes1 = {node.id():node for node in self.xrule1.rhs().nodes()}
        self.nodes2 = {node.id():node for node in self.xrule2.rhs().nodes()}

        # extract node labels and edge labels for use in testing
        self.nl_t = self.hmm.get_node_label("T")
        self.nl_w = self.hmm.get_node_label("W")
        self.el_x1 = self.hmm.get_edge_label("X")
        self.el_x2 = self.conjoinand.get_edge_label("X")

    def setUp(self):
        with open(os.path.join(os.path.dirname(__file__), 'hmm.json')) as f:
            self.hmm_json = json.load(f)
        with open(os.path.join(os.path.dirname(__file__), 'conjoinand.json')) as f:
            self.conjoinand_json = json.load(f)
        with open(os.path.join(os.path.dirname(__file__), 'conjunction.json')) as f:
            self.conjunction_json = json.load(f)        
        self.restore()

    def test_conjoinable(self):    
        self.assertTrue(conjoinable(self.xrule1, self.xrule2))        
        # rules have different numbers of nodes
        self.xrule1.rhs().add_node(fggs.Node(label=self.nl_t,id="v3"))
        self.assertFalse(conjoinable(self.xrule1, self.xrule2))
        self.restore()
        # rules have same number of nodes, different ids
        self.xrule1.rhs().add_node(fggs.Node(label=self.nl_t,id="v3"))
        self.xrule2.rhs().add_node(fggs.Node(label=self.nl_t,id="v4"))
        self.assertFalse(conjoinable(self.xrule1, self.xrule2))
        self.restore()
        # rules have same number of nodes, same ids, different labels
        self.xrule1.rhs().add_node(fggs.Node(label=self.nl_t,id="v3"))
        self.xrule2.rhs().add_node(fggs.Node(label=self.nl_w,id="v3"))
        self.assertFalse(conjoinable(self.xrule1, self.xrule2))
        self.restore()
        # rules have different numbers of nonterminals
        self.xrule1.rhs().add_edge(fggs.Edge(label=self.el_x1,nodes=[self.nodes1["v0"]],id="e4"))
        self.assertFalse(conjoinable(self.xrule1, self.xrule2))
        self.restore()
        # rules have same number of nonterminals, different ids
        self.xrule1.rhs().add_edge(fggs.Edge(label=self.el_x1,nodes=[self.nodes1["v0"]],id="e4"))
        self.xrule2.rhs().add_edge(fggs.Edge(label=self.el_x2,nodes=[self.nodes2["v0"]],id="e5"))
        self.assertFalse(conjoinable(self.xrule1, self.xrule2))
        self.restore()
        # rules have same number of nonterminals, same ids, different attachments
        self.xrule1.rhs().add_edge(fggs.Edge(label=self.el_x1,nodes=[self.nodes1["v0"]],id="e4"))
        self.xrule2.rhs().add_edge(fggs.Edge(label=self.el_x2,nodes=[self.nodes2["v1"]],id="e4"))
        self.assertFalse(conjoinable(self.xrule1, self.xrule2))
        self.restore()
        # rules have different external nodes
        self.xrule1.rhs().set_ext([self.nodes1["v1"]])
        self.assertFalse(conjoinable(self.xrule1, self.xrule2))
        self.restore()

    def test_conjunction(self):
        conjunction_check = conjoin_fggs(self.hmm, self.conjoinand)
        conjunction_json_check = formats.fgg_to_json(conjunction_check)
        self.maxDiff = 10000
        self.assertEqual(self.conjunction_json.keys(), conjunction_json_check.keys())
        self.assertEqual(self.conjunction_json['domains'], conjunction_json_check['domains'])
        self.assertEqual(self.conjunction_json['factors'], conjunction_json_check['factors'])
        self.assertEqual(self.conjunction_json['nonterminals'], conjunction_json_check['nonterminals'])
        self.assertEqual(self.conjunction_json['start'], conjunction_json_check['start'])

        # ignore order of rules
        for r in self.conjunction_json['rules']:
            self.assertTrue(r in conjunction_json_check['rules'], f'{r} {j_check["rules"]}')
        for r in conjunction_json_check['rules']:
            self.assertTrue(r in self.conjunction_json['rules'], r)
