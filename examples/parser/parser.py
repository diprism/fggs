import sys
import torch
import fggs
import trees
import argparse
import collections
import tqdm # type: ignore

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
            params[el] = torch.tensor(0., requires_grad=True)
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
    bigram_els = {
        (None, False) : fggs.EdgeLabel('start terminal', [nonterminal_nl, terminal_nl], is_terminal=True),
        (False, None) : fggs.EdgeLabel('terminal stop', [terminal_nl], is_terminal=True),
        (None, True) : fggs.EdgeLabel('start nonterminal', [nonterminal_nl, nonterminal_nl], is_terminal=True),
        (True, True) : fggs.EdgeLabel('nonterminal nonterminal', [nonterminal_nl, nonterminal_nl], is_terminal=True),
        (True, None) : fggs.EdgeLabel('nonterminal stop', [nonterminal_nl], is_terminal=True),
    }

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

        # One edge for each bigram of nodes (including START and STOP)
        hrhs.add_edge(fggs.Edge(bigram_els[None, pattern[0]], [parent, children[0]]))
        for i in range(len(pattern)-1):
            hrhs.add_edge(fggs.Edge(bigram_els[pattern[i], pattern[i+1]], [children[i], children[i+1]]))
        hrhs.add_edge(fggs.Edge(bigram_els[pattern[-1], None], [children[-1]]))
        
        hrhs.ext = [parent]
        hrg.add_rule(fggs.HRGRule(subtree_el, hrhs))

    for el in bigram_els.values():
        params[el] = torch.zeros(interp.shape(el), requires_grad=True)
        domains = [interp.domains[nl] for nl in el.type]
        interp.add_factor(el, fggs.CategoricalFactor(domains, params[el]))
        
else:
    print(f'unknown method: {args.method}', file=sys.stderr)
    exit(1)

hrg = fggs.factorize(hrg)
fgg = fggs.FGG(hrg, interp)

print('begin training')
# The learning rate should be set low enough that we don't easily jump out of the region where Z is finite.
opt = torch.optim.SGD(params.values(), lr=1e-3)

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
            w = torch.tensor(0.)
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
                            w += params[bigram_els[None, pattern[0]]][lhs_index, rhs_indices[0]]
                            for i in range(len(rhs)-1):
                                w += params[bigram_els[pattern[i], pattern[i+1]]][rhs_indices[i], rhs_indices[i+1]]
                            w += params[bigram_els[pattern[-1], None]][rhs_indices[-1]]

                        else:
                            assert False

            for el in params:
                interp.factors[el].weights = params[el]
                
            z = fggs.sum_product(fgg, method='newton', semiring=fggs.LogSemiring())

            loss = -w + len(minibatch) * z # type: ignore
            train_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            # Gradient clipping is crucial, since the gradient can have infinite components.
            # The clipping value should be high enough to quickly exit the region where Z is infinite.
            # The reciprocal of the learning rate seems to be a reasonable choice.
            torch.nn.utils.clip_grad_value_(params.values(), 1000.)
            opt.step()

            progress.update(len(minibatch))
    print(f'epoch={epoch+1} train_loss={train_loss}')
