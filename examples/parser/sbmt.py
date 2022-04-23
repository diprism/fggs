import sys
import math
import torch
import fggs
import trees
import argparse
import collections
import random
import tqdm # type: ignore

ap = argparse.ArgumentParser()
ap.add_argument('trainfile')
ap.add_argument('--device', dest="device", default="cpu")
args = ap.parse_args()

class ClipGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, val: float):
        ctx.val = val
        return inp

    @staticmethod
    def backward(ctx, grad_out):
        grad_in = torch.clip(grad_out, -ctx.val, ctx.val)
        return (grad_in, None)

# Read in training data, which consists of tokenized strings on the
# source side and Penn-Treebank-style trees on the target side.

traindata = []
svocab = set()
for line in open(args.trainfile):
    sline, tline = line.split('\t', 1)
    swords = sline.split()
    svocab.update(swords)
    ttree = trees.Tree.from_str(tline)
    if len(swords) == 0 or ttree is None: continue
    traindata.append((swords, ttree))
svocab.add('<BOS>')
svocab = {s:i for (i,s) in enumerate(svocab)}

### Create the encoder.

class Encoder(torch.nn.Module):
    def __init__(self, vocab, out_size):
        super().__init__()
        self.vocab = vocab
        dim = 512
        max_pos = 200
        self.word_embedding = torch.nn.Parameter(torch.empty((len(vocab), dim)))
        torch.nn.init.normal_(self.word_embedding)
        self.pos_encoding = torch.concat([
            torch.sin(torch.arange(max_pos).unsqueeze(1) / 10000**(2*torch.arange(dim//2)/dim)),
            torch.cos(torch.arange(max_pos).unsqueeze(1) / 10000**(2*torch.arange(dim//2)/dim)),
        ], dim=1)
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=dim, nhead=4),
            num_layers=4
        )
        self.out = torch.nn.Linear(dim, out_size)

    def forward(self, words):
        words = ['<BOS>'] + words
        words = [self.vocab[w] for w in words]
        words = self.word_embedding[words] + self.pos_encoding[:len(words),:]
        return self.out(self.transformer(words.unsqueeze(0))[0,0])

# Extract CFG rules from trees. We don't need to do any binarization
# or removal of unary rules. The fggs.factorize() routine does the
# equivalent of binarization for us, and the fggs.sum_product()
# routine is able to handle unary rules.

class Nonterminal(str):
    """A wrapper around strings to distinguish nonterminal symbols from
    terminal symbols."""
    def __repr__(self):
        return f'Nonterminal({repr(str(self))})'
    
cfg = collections.defaultdict(collections.Counter)
for _, tree in traindata:
    for node in tree.bottomup():
        if len(node.children) > 0:
            node.label = Nonterminal(node.label)
            cfg[node.label][tuple(child.label for child in node.children)] += 1

### Construct the FGG.

# The construction in the FGG paper (Appendix A) is not practical for
# a general CFG because the factor for a rule with k right-hand-side
# nonterminals is a tensor of size m^{k+1}, where m is the size of the
# nonterminal alphabet. Binarizing helps, but we implement two faster
# constructions below.

# Instead, this creates one FGG rule for each CFG rule. For
# example, the rule S -> NP VP becomes an FGG rule
#
#     S -> NP VP □
#
# where NP and VP are 0-ary nonterminal edges and □ is a 0-ary
# factor for the weight of the CFG rule.

fgg = fggs.FGG('TOP')

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
        fgg.new_rule(lhs, hrhs)

for el in rules.values():
    fgg.new_finite_factor(el, torch.tensor(0.))

### Factorize the FGG into smaller rules.

fgg = fggs.factorize_fgg(fgg)

### Train the model.

def minibatches(iterable, size=100):
    b = []
    for i, x in enumerate(iterable):
        if i % size == 0 and len(b) > 0:
            yield b
            b = []
        b.append(x)
    if len(b) > 0:
        yield b

encoder = Encoder(svocab, len(fgg.factors))
opt = torch.optim.Adam(encoder.parameters(), lr=3e-4)
for epoch in range(100):
    random.shuffle(traindata)
    train_loss = 0.
    with tqdm.tqdm(total=len(traindata)) as progress:
        for minibatch in minibatches(traindata, 1):
            loss = 0.
            for swords, ttree in minibatch:
                # The probability of a tree is the weight of a tree
                # divided by the total weight of all trees.

                weights = encoder(swords)
                for fi, fac in enumerate(fgg.factors.values()):
                    fac.weights = ClipGradient.apply(weights[fi], 10.)

                # Compute the weight of the tree.
            
                for node in ttree.bottomup():
                    if len(node.children) > 0:
                        lhs = node.label
                        rhs = tuple(child.label for child in node.children)
                        loss -= fgg.factors[rules[lhs, rhs]].weights

                # Compute the total weight of all trees.
                loss += fggs.sum_product(fgg, method='newton', semiring=fggs.LogSemiring(device=args.device))

            train_loss += loss.item()

            opt.zero_grad()
            loss.backward()

            # problem: grads are nan

            # If Z becomes infinite, some of the gradients can also
            # become infinite. So gradient clipping is crucial. The
            # clipping value should be high enough to quickly exit the
            # region where Z is infinite. The reciprocal of the
            # learning rate seems to be a reasonable choice.
            torch.nn.utils.clip_grad_value_(encoder.parameters(), 10.)
            
            opt.step()

            progress.update(len(minibatch))
    print(f'epoch={epoch+1} train_loss={train_loss}')
