from fgg_representation import NodeLabel, EdgeLabel, Node, Edge, FactorGraph, FGGRepresentation
from domain import FiniteDomain

# Define the node labels.
tagset = FiniteDomain("tagset", {"<s>", "Det", "N", "V", "</s>"})
vocab  = FiniteDomain("vocabulary", {"a", "the", "dog", "cat", "chased", "kicked"})
T = NodeLabel("T", tagset)
W = NodeLabel("W", vocab)

# Define the nonterminals.
S = EdgeLabel("S", False, (), None)
X = EdgeLabel("X", False, (T,), None)

# Define the terminals.
def make_constraint_function(value):
    def constraint_function(input_value):
        if input_value == value:
            return 1
        else:
            return 0
    return constraint_function

def transition_table(t1, t2):
    if (t1 == "<s>"):
        if (t2 == "Det"):
            return 1.0
    elif (t1 == "Det"):
        if (t2 == "N"):
            return 1.0
    elif (t1 == "N"):
        if (t2 == "V"):
            return 0.5
        elif (t2 == "</s>"):
            return 0.5
    elif (t1 == "V"):
        if (t2 == "N"):
            return 0.5
        elif (t2 == "</s>"):
            return 0.5
    return 0.0

def emission_probs(t, w):
    if (t == "Det"):
        if (w == "a") or (v == "the"):
            return 0.5
    elif (t == "N"):
        if (w == "cat") or (v == "dog"):
            return 0.5
    elif (t == "V"):
        if (w == "chased") or (v == "kicked"):
            return 0.5
    return 0.0

bos    = EdgeLabel("BOS", True, (T,), make_constraint_function("<s>"))
eos    = EdgeLabel("EOS", True, (T,), make_constraint_function("</s>"))
ttable = EdgeLabel("Transition table", True, (T, T), transition_table)
etable = EdgeLabel("Emission probabilities", True, (T, W), emission_probs)

# Define the FGG.
hmm = FGGRepresentation()
hmm.add_node_label(T)
hmm.add_node_label(W)
hmm.add_nonterminal(S)
hmm.add_nonterminal(X)
hmm.add_terminal(bos)
hmm.add_terminal(eos)
hmm.add_terminal(ttable)
hmm.add_terminal(etable)
hmm.set_start_symbol(S)

# Define the rules.
S_rhs_t1  = Node(T)
S_rhs_fac = Edge(bos, (S_rhs_t1,))
S_rhs_x2  = Edge(X, (S_rhs_t1,))

S_rhs = FactorGraph()
S_rhs.add_node(S_rhs_t1)
S_rhs.add_edge(S_rhs_fac)
S_rhs.add_edge(S_rhs_x2)

hmm.add_rule(S, S_rhs)

X_rhs1_t1   = Node(T)
X_rhs1_t2   = Node(T)
X_rhs1_w3   = Node(W)
X_rhs1_fac1 = Edge(ttable, (X_rhs1_t1, X_rhs1_t2))
X_rhs1_fac2 = Edge(etable, (X_rhs1_t2, X_rhs1_w3))
X_rhs1_x4   = Edge(X, (X_rhs1_t2,))

X_rhs1 = FactorGraph()
X_rhs1.add_ext_node(X_rhs1_t1)
X_rhs1.add_node(X_rhs1_t2)
X_rhs1.add_node(X_rhs1_w3)
X_rhs1.add_edge(X_rhs1_fac1)
X_rhs1.add_edge(X_rhs1_fac2)
X_rhs1.add_edge(X_rhs1_x4)

hmm.add_rule(X, X_rhs1)

X_rhs2_t1   = Node(T)
X_rhs2_t2   = Node(T)
X_rhs2_fac1 = Edge(ttable, (X_rhs2_t1, X_rhs2_t2))
X_rhs2_fac2 = Edge(bos, (X_rhs2_t2,))

X_rhs2 = FactorGraph()
X_rhs2.add_ext_node(X_rhs2_t1)
X_rhs2.add_node(X_rhs2_t2)
X_rhs2.add_edge(X_rhs2_fac1)
X_rhs2.add_edge(X_rhs2_fac2)

hmm.add_rule(X, X_rhs2)
