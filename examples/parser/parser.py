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

# Read in training data, which consists of Penn-Treebank-style trees.

traintrees = [trees.Tree.from_str(line) for line in open(args.trainfile)]

# Extract CFG rules from trees. We don't need to do any binarization
# or removal of unary rules. The fggs.factorize() routine does the
# equivalent of binarization for us, and the fggs.sum_product()
# routine is able to handle unary rules.

class Nonterminal(str):
    """A wrapper around strings to distinguish nonterminal symbols from
    terminal symbols."""
    def __repr__(self):
        return f'Nonterminal({repr(str(self))})'
    
cfg = collections.defaultdict(set)
for tree in traintrees:
    for node in tree.bottomup():
        if len(node.children) > 0:
            node.label = Nonterminal(node.label)
            cfg[node.label].add(tuple(child.label for child in node.children))

### Construct the FGG.

# The construction in the FGG paper (Appendix A) is not practical for
# a general CFG because the factor for a rule with k right-hand-side
# nonterminals is a tensor of size m^{k+1}, where m is the size of the
# nonterminal alphabet. Binarizing helps, but we implement two faster
# constructions below.

if args.method == 'rule':

    # The 'rule' method creates one FGG rule for each CFG rule. For
    # example, the rule S -> NP VP becomes an FGG rule
    #
    #     S -> NP VP □
    #
    # where NP and VP are 0-ary nonterminal edges and □ is a 0-ary
    # factor for the weight of the CFG rule.

    # Create the HRG.
    
    hrg = fggs.HRG('TOP')
    
    rules = {}
    for lhs in cfg:
        for rhs in cfg[lhs]:
            hrhs = fggs.Graph()
            
            # One nonterminal edge for each nonterminal on the rhs of
            # the CFG rule
            for x in rhs:
                if isinstance(x, Nonterminal):
                    hrhs.new_edge(x, [], is_nonterminal=True)
                    
            # One terminal edge for the weight of the CFG rule
            el = f'{repr(lhs)} -> {" ".join(map(repr, rhs))}'
            rules[lhs, rhs] = el # save for later use
            hrhs.new_edge(el, [], is_terminal=True)
            
            hrhs.ext = []
            hrg.new_rule(lhs, hrhs)

    # Make the HRG into an FGG.

    fgg = fggs.FGG(hrg)
    for el in rules.values():
        fgg.new_finite_factor(el, torch.tensor(0., requires_grad=True))

elif args.method == 'pattern':

    # The 'pattern' method is similar to Appendix A of the FGG paper,
    # generalized to arbitrary CFGs. It creates an FGG rule for every
    # pattern of terminals and nonterminals in the right-hand
    # side. For example, VP -> saw NP has the pattern "terminal
    # nonterminal". For this pattern, we create the FGG rule
    #   
    #                                    • nonterminal
    #                                   /
    #   nonterminal •                  □
    #               |    ->           / 
    #            subtree    terminal ∘---□---∘ nonterminal
    #                                |       |
    #                             subtree subtree
    #
    # where "subtree" is a nonterminal edge, the nodes labeled
    # "nonterminal" range over CFG nonterminals, and the node labeled
    # "terminal" ranges over CFG terminals.
    #
    # To reduce complexity, we don't create a single factor for the
    # whole rule; instead, we create a factor (□) for each bigram of
    # symbols. Specifically, one factor between the parent and
    # leftmost child, and one factor for each pair of neighboring
    # children.

    # Collect sets of CFG terminals, CFG nonterminals, and patterns.
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

    # Create the HRG.
    
    hrg = fggs.HRG('tree')

    # The starting rule just ensures that the root node is the CFG start symbol.
    hrhs = fggs.Graph()
    root = hrhs.new_node('nonterminal')
    hrhs.new_edge('is_start', [root], is_terminal=True)
    hrhs.new_edge('subtree', [root], is_nonterminal=True)
    hrg.new_rule('tree', hrhs)

    # Create an HRG rule for each pattern.
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

    # Make the HRG into an FGG.
    
    fgg = fggs.FGG(hrg)
    nonterminal_dom = fgg.new_finite_domain('nonterminal', nonterminals)
    terminal_dom = fgg.new_finite_domain('terminal', terminals)
    
    fgg.new_finite_factor(
        'is_start',
        torch.tensor([float(x == 'TOP') for x in nonterminals]))

    for el in hrg.terminals():
        if el.name != 'is_start':
            fgg.new_finite_factor(el.name, torch.zeros(fgg.shape(el), requires_grad=True))

else:
    print(f'unknown method: {args.method}', file=sys.stderr)
    exit(1)

### Factorize the FGG into smaller rules.

fgg.grammar = fggs.factorize(fgg.grammar)

### Train a globally-normalized model.

# This is fairly standard, except that we have to watch out for the
# possibility that the partition function becomes infinite.

# We can recover from this condition, but it will slow down training
# if it happens too much. So the learning rate should be set low
# enough that we don't easily jump out of the region where Z is
# finite.

params = [fac.weights for fac in fgg.factors.values() if fac.weights.requires_grad]
opt = torch.optim.SGD(params, lr=1e-3)

def minibatches(iterable, size=100):
    b = []
    for i, x in enumerate(iterable):
        if i % size == 0 and len(b) > 0:
            yield b
            b = []
        b.append(x)
    if len(b) > 0:
        yield b

for epoch in range(100):
    train_loss = 0.
    with tqdm.tqdm(total=len(traintrees)) as progress:
        for minibatch in minibatches(traintrees):

            # The probability of a tree is the weight of a tree
            # divided by the total weight of all trees.

            # Compute the weight of the tree.
            
            w = torch.tensor(0.)
            for tree in minibatch:
                for node in tree.bottomup():
                    if len(node.children) > 0:
                        lhs = node.label
                        rhs = tuple(child.label for child in node.children)
                        if args.method == 'rule':
                            w += fgg.factors[rules[lhs, rhs]].weights
                        elif args.method == 'pattern':
                            pattern = tuple(isinstance(x, Nonterminal) for x in rhs)
                            lhs_index = nonterminal_dom.numberize(lhs)
                            rhs_indices = tuple(nonterminal_dom.numberize(x) if isinstance(x, Nonterminal) else terminal_dom.numberize(x) for x in rhs)
                            w += fgg.factors[f'start {pattern[0]}'].weights[lhs_index, rhs_indices[0]]
                            for i in range(len(rhs)-1):
                                w += fgg.factors[f'{pattern[0]} {pattern[1]}'].weights[rhs_indices[0], rhs_indices[1]]
                            w += fgg.factors[f'{pattern[-1]} stop'].weights[rhs_indices[-1]]

                        else:
                            assert False

            # Compute the total weight of all trees.
            z = fggs.sum_product(fgg, method='newton', semiring=fggs.LogSemiring(device=args.device))

            loss = -w + len(minibatch) * z # type: ignore
            train_loss += loss.item()

            opt.zero_grad()
            loss.backward()

            # If Z becomes infinite, some of the gradients can also
            # become infinite. So gradient clipping is crucial. The
            # clipping value should be high enough to quickly exit the
            # region where Z is infinite. The reciprocal of the
            # learning rate seems to be a reasonable choice.
            torch.nn.utils.clip_grad_value_(params, 1000.)
            
            opt.step()

            progress.update(len(minibatch))
    print(f'epoch={epoch+1} train_loss={train_loss}')
