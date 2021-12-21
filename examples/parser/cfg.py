import torch
import fggs
import trees
import argparse
import collections

ap = argparse.ArgumentParser()
ap.add_argument('trainfile')
args = ap.parse_args()

# Extract CFG rules from trees. We don't do any binarization or removal of unary rules.

print('extract CFG')
cfgrules = collections.defaultdict(set)
traintrees = []
for line in open(args.trainfile):
    tree = trees.Tree.from_str(line)
    # Because the Treebank has nonterminals and terminals with the same name, we rename apart the nonterminals
    for node in tree.bottomup():
        if len(node.children) > 0:
            node.label = "nt_" + node.label
    traintrees.append(tree)
    for node in tree.bottomup():
        if len(node.children) > 0:
            cfgrules[node.label].add(tuple(child.label for child in node.children))

# There's more than one way to represent a CFG as an FGG.
# Here, each CFG rule becomes its own FGG rule.

print('convert to FGG')
interp = fggs.Interpretation()

pos = fggs.NodeLabel('pos')
pos_dom = fggs.FiniteDomain([None])
interp.add_domain(pos, pos_dom)

nts = {
    nt : fggs.EdgeLabel(nt, [pos, pos], is_nonterminal=True)
    for nt in cfgrules
}
hrg = fggs.HRG(nts['nt_TOP'])

params = {}
for lhs in cfgrules:
    for rhs in cfgrules[lhs]:
        hrhs = fggs.Graph()
        firstnode = prevnode = fggs.Node(pos)
        for x in rhs:
            newnode = fggs.Node(pos)
            if x in cfgrules: # if x is nonterminal
                el = nts[x]
                hrhs.add_edge(fggs.Edge(el, [prevnode, newnode]))
            prevnode = newnode
        el = fggs.EdgeLabel(f'{lhs} -> {" ".join(rhs)}', [], is_terminal=True)
        params[el] = torch.tensor(-3., requires_grad=True)
        interp.add_factor(el, fggs.CategoricalFactor([], 0.)) # will set weight later
        hrhs.add_edge(fggs.Edge(el, []))
        hrhs.ext = [firstnode, prevnode]
        hrule = fggs.HRGRule(nts[lhs], hrhs)
        hrg.add_rule(hrule)

hrg = fggs.factorize(hrg)
fgg = fggs.FGG(hrg, interp)

opt = torch.optim.SGD(params.values(), lr=1e-2)

for epoch in range(100):
    train_loss = 0.
    for tree in traintrees:
        for el, fac in interp.factors.items():
            if el in params:
                fac._weights = torch.exp(params[el])

        w = 0.
        for node in tree.bottomup():
            if len(node.children) > 0:
                lhs = node.label
                rhs = [child.label for child in node.children]
                el = fggs.EdgeLabel(f'{lhs} -> {" ".join(rhs)}', [], is_terminal=True)
                w += params[el]

        z = fggs.sum_product(fgg, method='fixed-point')

        loss = -w + torch.log(z)
        train_loss += loss.item()
        print(f'p(tree) = {w}, Z = {z.item()}')

        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f'epoch={epoch+1} train_loss={train_loss}, Z={z.item()}')

