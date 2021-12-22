import torch
import fggs
import trees
import argparse
import collections

ap = argparse.ArgumentParser()
ap.add_argument('trainfile')
args = ap.parse_args()

# Read in training data
print('read training data')
traintrees = [trees.Tree.from_str(line) for line in open(args.trainfile)]

# Extract CFG rules from trees. We don't do any binarization or removal of unary rules.
print('extract CFG')
cfg = collections.defaultdict(set)
nts = {}
pos = fggs.NodeLabel('pos')
for tree in traintrees:
    # Because the Treebank has nonterminals and terminals with the same name, we change the nonterminals to EdgeLabels
    for node in tree.bottomup():
        if len(node.children) > 0:
            if node.label not in nts:
                nts[node.label] = fggs.EdgeLabel(node.label, [pos, pos], is_nonterminal=True)
            node.label = nts[node.label]
    for node in tree.bottomup():
        if len(node.children) > 0:
            cfg[node.label].add(tuple(child.label for child in node.children))

# There's more than one way to represent a CFG as an FGG.
# Here, each CFG rule becomes its own FGG rule.

print('convert to FGG')

interp = fggs.Interpretation()

pos_dom = fggs.FiniteDomain([None])
interp.add_domain(pos, pos_dom)

hrg = fggs.HRG(nts['TOP'])

rules = {}
params = {}
for lhs in cfg:
    for rhs in cfg[lhs]:
        hrhs = fggs.Graph()
        firstnode = prevnode = fggs.Node(pos)
        for x in rhs:
            newnode = fggs.Node(pos)
            if x in cfg: # if x is nonterminal
                hrhs.add_edge(fggs.Edge(x, [prevnode, newnode]))
            prevnode = newnode
        el = fggs.EdgeLabel(f'{lhs} -> {" ".join(map(str,rhs))}', [], is_terminal=True)
        rules[lhs, rhs] = el
        params[el] = torch.tensor(-5., requires_grad=True)
        interp.add_factor(el, fggs.CategoricalFactor([], 0.)) # will set weight later
        hrhs.add_edge(fggs.Edge(el, []))
        hrhs.ext = [firstnode, prevnode]
        hrule = fggs.HRGRule(lhs, hrhs)
        hrg.add_rule(hrule)

hrg = fggs.factorize(hrg)
fgg = fggs.FGG(hrg, interp)

opt = torch.optim.SGD(params.values(), lr=1e-2)

for epoch in range(100):
    train_loss = 0.
    for tree in traintrees:
        for el in params:
            interp.factors[el]._weights = torch.exp(params[el])

        w = 0.
        for node in tree.bottomup():
            if len(node.children) > 0:
                lhs = node.label
                rhs = tuple(child.label for child in node.children)
                w += params[rules[lhs, rhs]]

        z = fggs.sum_product(fgg, method='fixed-point', kmax=10, tol=1e-20)

        loss = -w + torch.log(z)
        train_loss += loss.item()

        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f'epoch={epoch+1} train_loss={train_loss}, Z={z.log().item()}')

