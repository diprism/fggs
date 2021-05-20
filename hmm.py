from fgg_representation import NodeLabel, EdgeLabel, Node, Edge, FactorGraph, FGGRule, FGGRepresentation
from domains import FiniteDomain
from factors import CategoricalFactor

# Define the node labels.
tagset = FiniteDomain(["<s>", "Det", "N", "V", "</s>"])
vocab  = FiniteDomain(["a", "the", "dog", "cat", "chased", "kicked"])
T = NodeLabel("T", tagset)
W = NodeLabel("W", vocab)

# Define the nonterminals.
S = EdgeLabel("S", False, ())
X = EdgeLabel("X", False, (T,))

# Define the terminals.
def make_constraint_factor(domain, value):
    weights = [0.] * domain.size()
    weights[domain.numberize(value)] = 1.
    return CategoricalFactor([domain], weights)

bos    = EdgeLabel("BOS", True, (T,), make_constraint_factor(tagset, "<s>"))
eos    = EdgeLabel("EOS", True, (T,), make_constraint_factor(tagset, "</s>"))
ttable = EdgeLabel("Ttable", True, (T, T),
                   CategoricalFactor([tagset, tagset],
                                     [[0, 1, 0,   0,   0],
                                      [0, 0, 1,   0,   0],
                                      [0, 0, 0,   0.5, 0.5],
                                      [0, 0, 0.5, 0,   0.5],
                                      [0, 0, 0,   0,   0]]))
etable = EdgeLabel("Etable", True, (T, W),
                   CategoricalFactor([tagset, vocab],
                                     [[0,   0,   0,   0,   0,   0],
                                      [0.5, 0.5, 0,   0,   0,   0],
                                      [0,   0,   0.5, 0.5, 0,   0],
                                      [0,   0,   0,   0,   0.5, 0.5],
                                      [0,   0,   0,   0,   0,   0]]))

# Define the FGG.
hmm = FGGRepresentation()
hmm.add_node_label(T)
print(T)
hmm.add_node_label(W)
print(W)
hmm.add_nonterminal(S)
print(S)
hmm.add_nonterminal(X)
print(X)
hmm.add_terminal(bos)
print(bos)
hmm.add_terminal(eos)
print(eos)
hmm.add_terminal(ttable)
print(ttable)
hmm.add_terminal(etable)
print(etable)
hmm.set_start_symbol(S)
print()

# Define the rules.
S_rhs_t1  = Node(T)
print(S_rhs_t1)
S_rhs_fac = Edge(bos, (S_rhs_t1,))
print(S_rhs_fac)
S_rhs_x2  = Edge(X, (S_rhs_t1,))
print(S_rhs_x2)

S_rhs = FactorGraph()
S_rhs.add_node(S_rhs_t1)
S_rhs.add_edge(S_rhs_fac)
S_rhs.add_edge(S_rhs_x2)
print()
print(S_rhs)

S_rule1 = FGGRule(S, S_rhs)
print()
print(S_rule1)
hmm.add_rule(S_rule1)

X_rhs1_t1   = Node(T)
X_rhs1_t2   = Node(T)
X_rhs1_w3   = Node(W)
X_rhs1_fac1 = Edge(ttable, (X_rhs1_t1, X_rhs1_t2))
X_rhs1_fac2 = Edge(etable, (X_rhs1_t2, X_rhs1_w3))
X_rhs1_x4   = Edge(X, (X_rhs1_t2,))

X_rhs1 = FactorGraph()
X_rhs1.add_node(X_rhs1_t1)
X_rhs1.add_node(X_rhs1_t2)
X_rhs1.add_node(X_rhs1_w3)
X_rhs1.add_edge(X_rhs1_fac1)
X_rhs1.add_edge(X_rhs1_fac2)
X_rhs1.add_edge(X_rhs1_x4)
X_rhs1.set_ext([X_rhs1_t1])

X_rule1 = FGGRule(X, X_rhs1)
hmm.add_rule(X_rule1)

X_rhs2_t1   = Node(T)
X_rhs2_t2   = Node(T)
X_rhs2_fac1 = Edge(ttable, (X_rhs2_t1, X_rhs2_t2))
X_rhs2_fac2 = Edge(eos, (X_rhs2_t2,))

X_rhs2 = FactorGraph()
X_rhs2.add_node(X_rhs2_t1)
X_rhs2.add_node(X_rhs2_t2)
X_rhs2.add_edge(X_rhs2_fac1)
X_rhs2.add_edge(X_rhs2_fac2)
X_rhs2.set_ext([X_rhs2_t1])

X_rule2 = FGGRule(X, X_rhs2)
hmm.add_rule(X_rule2)
print()
print(hmm)
