import torch
import fggs
import trees
import argparse
import collections
import tqdm

ap = argparse.ArgumentParser()
ap.add_argument('trainfile')
ap.add_argument('-m', dest="method", default="rule", help="Method for converting CFG to FGG ('rule' or 'pattern')")
ap.add_argument('-b', dest="binarize", default=False, action="store_true", help="Binarize trees")
args = ap.parse_args()

if args.method == 'pattern':
    args.binarize = True

# Read in training data
print('read training data')
traintrees = [trees.Tree.from_str(line) for line in open(args.trainfile)]

def binarize(node):
    children = [binarize(child) for child in node.children]
    if len(children) <= 2:
        return trees.Node(node.label, children)
    new = children[-1]
    for child in reversed(children[1:-1]):
        new = trees.Node(f'{child.label}+{new.label}', [child, new])

    return trees.Node(node.label, [children[0], new])

if args.binarize:
    for tree in traintrees:
        tree.root = binarize(tree.root)

# Extract CFG rules from trees. We don't do any binarization or removal of unary rules.
print('extract CFG')
cfg = collections.defaultdict(set)
class Nonterminal(str):
    def __repr__(self):
        return f'Nonterminal({repr(str(self))})'
for tree in traintrees:
    for node in tree.bottomup():
        if len(node.children) > 0:
            node.label = Nonterminal(node.label)
            cfg[node.label].add(tuple(child.label for child in node.children))

print('convert to FGG')

interp = fggs.Interpretation()
params = {}

if args.method == 'rule':
    rules = {}
    
    def edgelabel(name):
        return fggs.EdgeLabel(name, [], is_nonterminal=True)

    hrg = fggs.HRG(edgelabel('TOP'))

    for lhs in cfg:
        for rhs in cfg[lhs]:
            hrhs = fggs.Graph()
            for x in rhs:
                if isinstance(x, Nonterminal):
                    hrhs.add_edge(fggs.Edge(edgelabel(x), []))
            el = fggs.EdgeLabel(f'{repr(lhs)} -> {" ".join(map(repr, rhs))}', [], is_terminal=True)
            rules[lhs, rhs] = el
            params[el] = torch.tensor(-5., requires_grad=True)
            interp.add_factor(el, fggs.CategoricalFactor([], 0.)) # will set weight later
            hrhs.add_edge(fggs.Edge(el, []))
            hrhs.ext = []
            hrule = fggs.HRGRule(edgelabel(lhs), hrhs)
            hrg.add_rule(hrule)

elif args.method == 'pattern':
    for lhs in cfg:
        patterns = set()
        nonterminals = set()
        terminals = set()
        for lhs in cfg:
            nonterminals.add(lhs)
            for rhs in cfg[lhs]:
                patterns.add(tuple(isinstance(x, Nonterminal) for x in rhs))
                for x in rhs:
                    if isinstance(x, Nonterminal):
                        nonterminals.add(x)
                    else:
                        terminals.add(x)
                        
    nonterminal_nl = fggs.NodeLabel('nonterminal')
    nonterminal_dom = fggs.FiniteDomain(nonterminals)
    interp.add_domain(nonterminal_nl, nonterminal_dom)
    terminal_nl = fggs.NodeLabel('terminal')
    terminal_dom = fggs.FiniteDomain(terminals)
    interp.add_domain(terminal_nl, terminal_dom)
    
    tree_el = fggs.EdgeLabel('tree', [], is_nonterminal=True)
    subtree_el = fggs.EdgeLabel('subtree', [nonterminal_nl], is_nonterminal=True)
    pattern_els = {}

    hrg = fggs.HRG(tree_el)
    hrhs = fggs.Graph()
    root = fggs.Node(nonterminal_nl)
    hrhs.add_node(root)
    el = fggs.EdgeLabel('is_start', [nonterminal_nl], is_terminal=True)
    hrhs.add_edge(fggs.Edge(el, [root]))
    weights = torch.tensor([x == 'TOP' for x in nonterminal_dom.values], dtype=torch.get_default_dtype())
    interp.add_factor(el, fggs.CategoricalFactor([nonterminal_dom], weights))
    hrhs.add_edge(fggs.Edge(subtree_el, [root]))
    hrg.add_rule(fggs.HRGRule(tree_el, hrhs))

    for pattern in patterns:
        hrhs = fggs.Graph()
        parent = fggs.Node(nonterminal_nl)
        hrhs.add_node(parent)
        children = []
        for is_nonterminal in pattern:
            if is_nonterminal:
                child = fggs.Node(nonterminal_nl)
                hrhs.add_node(child)
                hrhs.add_edge(fggs.Edge(subtree_el, [child]))
            else:
                child = fggs.Node(terminal_nl)
                hrhs.add_node(child)
            children.append(child)
        el = fggs.EdgeLabel(
            ' '.join(child.label.name for child in children),
            [parent.label]+[child.label for child in children],
            is_terminal=True
        )
        pattern_els[pattern] = el
        domains = [interp.domains[nl] for nl in el.type]
        shape = [dom.size() for dom in domains]
        params[el] = torch.full(shape, fill_value=-10., requires_grad=True)
        weights = torch.zeros(shape) # will set weights later
        interp.add_factor(el, fggs.CategoricalFactor(domains, weights)) 
        hrhs.add_edge(fggs.Edge(el, [parent]+children))
        hrhs.ext = [parent]

        hrg.add_rule(fggs.HRGRule(subtree_el, hrhs))

else:
    print(f'unknown method: {args.method}', file=sys.stderr)
    exit(1)

hrg = fggs.factorize(hrg)
fgg = fggs.FGG(hrg, interp)

print('begin training')
opt = torch.optim.Adam(params.values(), lr=1e-1)

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
            w = 0.
            for tree in minibatch:
                for node in tree.bottomup():
                    if len(node.children) > 0:
                        lhs = node.label
                        rhs = tuple(child.label for child in node.children)
                        if args.method == 'rule':
                            w += params[rules[lhs, rhs]]
                        elif args.method == 'pattern':
                            pattern = tuple(isinstance(x, Nonterminal) for x in rhs)
                            lhs_index = nonterminal_dom.numberize(lhs)
                            rhs_indices = tuple(nonterminal_dom.numberize(x) if isinstance(x, Nonterminal) else terminal_dom.numberize(x) for x in rhs)
                            w += params[pattern_els[pattern]][(lhs_index,)+rhs_indices]
                        else:
                            assert False

            # PyTorch doesn't allow reusing non-leaf nodes,
            # so we have to recompute exps every time
            for el in params:
                interp.factors[el].weights = torch.exp(params[el])
                
            # Newton's method currently works better than fixed-point iteration
            # for avoiding z = 0.
            z = fggs.sum_product(fgg, method='newton', kmax=100, tol=1e-30)

            loss = -w + len(minibatch) * torch.log(z)
            train_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            progress.update(len(minibatch))
    print(f'epoch={epoch+1} train_loss={train_loss}')
