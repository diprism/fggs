import torch
import fggs
import trees
import argparse
import collections
import tqdm

ap = argparse.ArgumentParser()
ap.add_argument('trainfile')
ap.add_argument('-m', dest="method", default="rule", help="Method for converting CFG to FGG ('rule' or 'pattern')")
args = ap.parse_args()

# Read in training data
print('read training data')
traintrees = [trees.Tree.from_str(line) for line in open(args.trainfile)]

# Extract CFG rules from trees. We don't do any binarization or removal of unary rules.
print('extract CFG')
cfg = collections.defaultdict(set)
nts = {}
for tree in traintrees:
    # Because the Treebank has nonterminals and terminals with the same name, we change the nonterminals to EdgeLabels
    for node in tree.bottomup():
        if len(node.children) > 0:
            if node.label not in nts:
                nts[node.label] = fggs.EdgeLabel(node.label, [], is_nonterminal=True)
            node.label = nts[node.label]
    for node in tree.bottomup():
        if len(node.children) > 0:
            cfg[node.label].add(tuple(child.label for child in node.children))

print('convert to FGG')

interp = fggs.Interpretation()
if args.method == 'rule':
    hrg = fggs.HRG(nts['TOP'])

    rules = {}
    params = {}
    for lhs in cfg:
        for rhs in cfg[lhs]:
            hrhs = fggs.Graph()
            for x in rhs:
                if x in cfg: # if x is nonterminal
                    hrhs.add_edge(fggs.Edge(x, []))
            el = fggs.EdgeLabel(f'{lhs.name} -> {" ".join([y.name if isinstance(y, fggs.EdgeLabel) else y for y in rhs])}', [], is_terminal=True)
            rules[lhs, rhs] = el
            params[el] = torch.tensor(-5., requires_grad=True)
            interp.add_factor(el, fggs.CategoricalFactor([], 0.)) # will set weight later
            hrhs.add_edge(fggs.Edge(el, []))
            hrhs.ext = []
            hrule = fggs.HRGRule(lhs, hrhs)
            hrg.add_rule(hrule)

elif args.method == 'pattern':
    raise NotImplementedError()

else:
    print(f'unknown method: {args.method}', file=sys.stderr)
    exit(1)

hrg = fggs.factorize(hrg)
fgg = fggs.FGG(hrg, interp)

print('begin training')
opt = torch.optim.SGD(params.values(), lr=1e-2)

def minibatches(iterable, size):
    b = []
    for i, x in enumerate(iterable):
        if i % size == 0 and len(b) > 0:
            yield b
            b = []
        b.append(x)
    if len(b) > 0:
        yield b

minibatch_size = 100

for epoch in range(100):
    train_loss = 0.
    with tqdm.tqdm(total=len(traintrees)) as progress:
        for minibatch in minibatches(traintrees, minibatch_size):
            for el in params:
                interp.factors[el].weights = torch.exp(params[el])
            w = 0.
            for tree in minibatch:
                for node in tree.bottomup():
                    if len(node.children) > 0:
                        lhs = node.label
                        rhs = tuple(child.label for child in node.children)
                        w += params[rules[lhs, rhs]]

            # Newton's method currently works better than fixed-point iteration
            # for avoiding z = 0.
            z = fggs.sum_product(fgg, method='newton', kmax=100, tol=1e-30)

            loss = -w + minibatch_size * torch.log(z)
            train_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            progress.update(len(minibatch))
    print(f'epoch={epoch+1} train_loss={train_loss}')
