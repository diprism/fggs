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
    
    hrg = fggs.HRG('TOP')
    fgg = fggs.FGG(hrg, interp)

    for lhs in cfg:
        for rhs in cfg[lhs]:
            el = f'{repr(lhs)} -> {" ".join(map(repr, rhs))}'
            rules[lhs, rhs] = el
            
            hrhs = fggs.Graph()
            for x in rhs:
                if isinstance(x, Nonterminal):
                    hrhs.new_edge(x, [], is_nonterminal=True)
            hrhs.new_edge(el, [], is_terminal=True)
            hrhs.ext = []
            hrg.new_rule(lhs, hrhs)
            
            params[el] = torch.tensor(0., requires_grad=True)
            fgg.new_categorical_factor(el, 0.) # will set weight later

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
                        
    hrg = fggs.HRG('tree')
    fgg = fggs.FGG(hrg, interp)
    
    nonterminal_dom = fgg.new_finite_domain('nonterminal', nonterminals)
    terminal_dom = fgg.new_finite_domain('terminal', terminals)
    
    pattern_els = {}

    hrhs = fggs.Graph()
    root = hrhs.new_node('nonterminal')
    hrhs.new_edge('is_start', [root], is_terminal=True)
    hrhs.new_edge('subtree', [root], is_nonterminal=True)
    hrg.new_rule('tree', hrhs)
    weights = torch.tensor([x == 'TOP' for x in nonterminals], dtype=torch.get_default_dtype())
    fgg.new_categorical_factor('is_start', weights)

    for pattern in patterns:
        hrhs = fggs.Graph()
        parent = hrhs.new_node('nonterminal')
        children = []
        for is_nonterminal in pattern:
            if is_nonterminal:
                child = hrhs.new_node('nonterminal')
                hrhs.new_edge('subtree', [child], is_nonterminal=True)
            else:
                child = hrhs.new_node('terminal')
            children.append(child)
        el = ' '.join(child.label.name for child in children)
        pattern_els[pattern] = el
        hrhs.new_edge(el, [parent]+children, is_terminal=True)
        hrhs.ext = [parent]
        hrg.new_rule('subtree', hrhs)
        
        shape = (len(nonterminals),) + tuple(len(nonterminals) if p else len(terminals) for p in pattern)
        params[el] = torch.zeros(shape, requires_grad=True)
        weights = torch.zeros(shape) # will set weights later
        fgg.new_categorical_factor(el, weights)

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
                            w += params[pattern_els[pattern]][(lhs_index,)+rhs_indices]
                        else:
                            assert False

            for el in params:
                interp.factors[fgg.grammar.get_edge_label(el)].weights = params[el]
                
            z = fggs.sum_product(fgg, method='fixed-point', semiring=fggs.LogSemiring())

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
