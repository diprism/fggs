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
ap.add_argument('--device', dest="device", default="cpu")
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

if args.method == 'rule':
    hrg = fggs.HRG('TOP')
    
    rules = {}
    for lhs in cfg:
        for rhs in cfg[lhs]:
            hrhs = fggs.Graph()
            
            # One nonterminal edge for each CFG nonterminal
            for x in rhs:
                if isinstance(x, Nonterminal):
                    hrhs.new_edge(x, [], is_nonterminal=True)
                    
            # One terminal edge for each rule
            el = f'{repr(lhs)} -> {" ".join(map(repr, rhs))}'
            rules[lhs, rhs] = el # save for later use
            hrhs.new_edge(el, [], is_terminal=True)
            
            hrhs.ext = []
            hrg.new_rule(lhs, hrhs)

    interp = fggs.Interpretation()
    fgg = fggs.FGG(hrg, interp)
    for el in rules.values():
        fgg.new_categorical_factor(el, torch.tensor(0., requires_grad=True))

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
    
    hrhs = fggs.Graph()
    root = hrhs.new_node('nonterminal')
    hrhs.new_edge('is_start', [root], is_terminal=True)
    hrhs.new_edge('subtree', [root], is_nonterminal=True)
    hrg.new_rule('tree', hrhs)

    for pattern in patterns:
        hrhs = fggs.Graph()

        # A node for each child, and a nonterminal edge for each CFG nonterminal
        parent = hrhs.new_node('nonterminal')
        children = []
        for is_nonterminal in pattern:
            if is_nonterminal:
                child = hrhs.new_node('nonterminal')
                hrhs.new_edge('subtree', [child], is_nonterminal=True)
            else:
                child = hrhs.new_node('terminal')
            children.append(child)

        # One edge for each bigram of nodes (including START and STOP)
        hrhs.new_edge(f'start {pattern[0]}', [parent, children[0]], is_terminal=True)
        for i in range(len(pattern)-1):
            hrhs.new_edge(f'{pattern[i]} {pattern[i+1]}', [children[i], children[i+1]], is_terminal=True)
        hrhs.new_edge(f'{pattern[-1]} stop', [children[-1]], is_terminal=True)
        
        hrhs.ext = [parent]
        hrg.new_rule('subtree', hrhs)

    interp = fggs.Interpretation()
    fgg = fggs.FGG(hrg, interp)
    nonterminal_dom = fgg.new_finite_domain('nonterminal', nonterminals)
    terminal_dom = fgg.new_finite_domain('terminal', terminals)
    
    fgg.new_categorical_factor(
        'is_start',
        torch.tensor([float(x == 'TOP') for x in nonterminals]))

    for el in hrg.terminals():
        if el.name != 'is_start':
            fgg.new_categorical_factor(el.name, torch.zeros(interp.shape(el), requires_grad=True))

else:
    print(f'unknown method: {args.method}', file=sys.stderr)
    exit(1)

hrg = fggs.factorize(hrg)
fgg = fggs.FGG(hrg, interp)

print('begin training')
# The learning rate should be set low enough that we don't easily jump out of the region where Z is finite.
params = [fac.weights for fac in interp.factors.values() if fac.weights.requires_grad]
opt = torch.optim.SGD(params, lr=1e-3)

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
                            w += interp.factors[hrg.get_edge_label(rules[lhs, rhs])].weights
                        elif args.method == 'pattern':
                            pattern = tuple(isinstance(x, Nonterminal) for x in rhs)
                            lhs_index = nonterminal_dom.numberize(lhs)
                            rhs_indices = tuple(nonterminal_dom.numberize(x) if isinstance(x, Nonterminal) else terminal_dom.numberize(x) for x in rhs)
                            w += interp.factors[hrg.get_edge_label(f'start {pattern[0]}')].weights[lhs_index, rhs_indices[0]]
                            for i in range(len(rhs)-1):
                                w += interp.factors[hrg.get_edge_label(f'{pattern[0]} {pattern[1]}')].weights[rhs_indices[0], rhs_indices[1]]
                            w += interp.factors[hrg.get_edge_label(f'{pattern[-1]} stop')].weights[rhs_indices[-1]]

                        else:
                            assert False
                
            z = fggs.sum_product(fgg, method='newton', semiring=fggs.LogSemiring(device=args.device))

            loss = -w + len(minibatch) * z # type: ignore
            train_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            # Gradient clipping is crucial, since the gradient can have infinite components.
            # The clipping value should be high enough to quickly exit the region where Z is infinite.
            # The reciprocal of the learning rate seems to be a reasonable choice.
            torch.nn.utils.clip_grad_value_(params, 1000.)
            opt.step()

            progress.update(len(minibatch))
    print(f'epoch={epoch+1} train_loss={train_loss}')
