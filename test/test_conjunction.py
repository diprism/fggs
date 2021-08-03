import unittest
import json
import os
import fggs
from fggs.conjunction import check_namespace_collisions, nonterminal_pairs, conjoinable, conjoin_rules, conjoin_fggs


class TestConjunction(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(__file__), 'hmm.json')) as f:
            self.hmm_json = json.load(f)
        with open(os.path.join(os.path.dirname(__file__), 'conjunct.json')) as f:
            self.conjunct_json = json.load(f)
        with open(os.path.join(os.path.dirname(__file__), 'conjunction.json')) as f:
            self.conjunction_json = json.load(f)        
        self.restore()

    def restore(self):
        self.hmm = fggs.json_to_fgg(self.hmm_json)
        self.conjunct = fggs.json_to_fgg(self.conjunct_json)
        self.conjunction = fggs.json_to_fgg(self.conjunction_json)
        
        # extract specific rules for testing conjunction
        xrules1 = self.hmm.rules(self.hmm.get_nonterminal("X"))
        for rule in xrules1:
            if len(rule.rhs().nodes()) == 3:
                self.xrule1 = rule
        
        xrules2 = self.conjunct.rules(self.conjunct.get_nonterminal("X"))
        for rule in xrules2:
            if "Y" in [nt.label.name for nt in rule.rhs().nonterminals()]:
                self.xrule2 = rule
        
        xrules3 = self.conjunction.rules(self.conjunction.get_nonterminal("<X,X>"))
        for rule in xrules3:
            if "Y" in [nt.label.name for nt in rule.rhs().nonterminals()]:
                self.xrule3 = rule
        
        # get all the nodes for easy access
        self.nodes1 = {node.id:node for node in self.xrule1.rhs().nodes()}
        self.nodes2 = {node.id:node for node in self.xrule2.rhs().nodes()}

        # extract node labels and edge labels for use in testing
        self.nl_t = self.hmm.get_node_label("T")
        self.nl_w = self.hmm.get_node_label("W")
        self.el_x1 = self.hmm.get_edge_label("X")
        self.el_x2 = self.conjunct.get_edge_label("X")

    def test_check_namespace_collisions(self):
        dom1 = fggs.FiniteDomain([5])
        nl1 = fggs.NodeLabel("collide_nl", dom1)
        nl2 = fggs.NodeLabel("collide_nl", fggs.FiniteDomain([6]))
        el1 = fggs.EdgeLabel("collide_el", [nl1], fggs.ConstantFactor([dom1], 7))
        el2 = fggs.EdgeLabel("collide_el", [nl1])
        nl3 = fggs.NodeLabel("don't_collide", fggs.FiniteDomain([8]))
        el3 = fggs.EdgeLabel("don't_collide", [nl1])
        self.hmm.add_node_label(nl1)
        self.conjunct.add_node_label(nl2)
        self.hmm.add_terminal(el1)
        self.conjunct.add_nonterminal(el2)
        self.hmm.add_node_label(nl3)
        self.conjunct.add_nonterminal(el3)
        (n, e) = check_namespace_collisions(self.hmm, self.conjunct)
        self.assertTrue(len(n) == 1)
        self.assertTrue(len(e) == 1)
        with self.assertRaises(ValueError):
            conjoin_fggs(self.hmm, self.conjunct)

    def test_nonterminal_pairs(self):
        fgg1 = fggs.FGG()
        fgg2 = fggs.FGG()
        fgg1.add_nonterminal(fggs.EdgeLabel(name="X", is_terminal=False, node_labels=()))
        fgg2.add_nonterminal(fggs.EdgeLabel(name="Y,Z", is_terminal=False, node_labels=()))
        fgg1.add_nonterminal(fggs.EdgeLabel(name="X,Y", is_terminal=False, node_labels=()))
        fgg2.add_nonterminal(fggs.EdgeLabel(name="Z", is_terminal=False, node_labels=()))
        nt_map = nonterminal_pairs(fgg1, fgg2)
        self.assertEqual(sorted(nt.name for nt in nt_map.values()), ["<X,Y,Y,Z>", "<X,Y,Z>", "<X,Y,Z>_2", "<X,Z>"])

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
        conjunction_check = conjoin_fggs(self.hmm, self.conjunct)
        conjunction_json_check = fggs.fgg_to_json(conjunction_check)
        self.maxDiff = 10000
        self.assertEqual(self.conjunction_json.keys(), conjunction_json_check.keys())
        self.assertEqual(self.conjunction_json['domains'], conjunction_json_check['domains'])
        self.assertEqual(self.conjunction_json['factors'], conjunction_json_check['factors'])
        self.assertEqual(self.conjunction_json['nonterminals'], conjunction_json_check['nonterminals'])
        self.assertEqual(self.conjunction_json['start'], conjunction_json_check['start'])

        # ignore order of rules
        for r in self.conjunction_json['rules']:
            self.assertTrue(r in conjunction_json_check['rules'])
        for r in conjunction_json_check['rules']:
            self.assertTrue(r in self.conjunction_json['rules'])
