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
ap.add_argument('-b', dest="minibatch_size", type=int, default=1)
args = ap.parse_args()

class ClipGradValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, val: float):
        ctx.val = val
        return inp

    @staticmethod
    def backward(ctx, grad):
        if torch.any(grad.isinf()):
            grad = grad.nan_to_num()
            grad /= grad.abs().max()
            grad *= ctx.val
        return (grad, None)
clip_grad_value = ClipGradValue.apply

# Read in training data, which consists of tokenized strings on the
# source side and Penn-Treebank-style trees on the target side.

def annotate(tree):
    def visit(node):
        if node.label == 'NP':
            l = 1
        else:
            l = 0
        for child in node.children:
            visit(child)
            l += child.length
        node.length = l
        if node.label != 'TOP':
            node.label += f'-{l}'
        return node
    return visit(tree.root)
            
traindata = []
svocab = set()
for line in open(args.trainfile):
    sline, tline = line.split('\t', 1)
    swords = sline.split()
    svocab.update(swords)
    ttree = trees.Tree.from_str(tline)
    if len(swords) == 0 or ttree is None: continue
    ttree = annotate(ttree)
    traindata.append((swords, ttree))
svocab.add('<BOS>')
svocab.add('<PAD>')
svocab = {s:i for (i,s) in enumerate(svocab)}

### Create the encoder.

class TransformerEncoderLayer(torch.nn.TransformerEncoderLayer):
    """Same as torch.nn.TransformerEncoderLayer, but with residual connections after layer normalization."""
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, self.last_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask)
        src2 = self.norm1(src2)
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = self.norm2(src2)
        src = src + self.dropout2(src2)
        return src
    
class Encoder(torch.nn.Module):
    def __init__(self, vocab, out_size):
        super().__init__()
        self.vocab = vocab
        dim = 256
        max_pos = 200
        self.word_embedding = torch.nn.Parameter(torch.empty((len(vocab), dim)))
        torch.nn.init.normal_(self.word_embedding)
        self.pos_encoding = torch.concat([
            torch.sin(torch.arange(max_pos).unsqueeze(1) / 10000**(2*torch.arange(dim//2)/dim)),
            torch.cos(torch.arange(max_pos).unsqueeze(1) / 10000**(2*torch.arange(dim//2)/dim)),
        ], dim=1)
        self.transformer = torch.nn.TransformerEncoder(
            TransformerEncoderLayer(d_model=dim, nhead=4),
            num_layers=4
        )
        self.out = torch.nn.Linear(dim, out_size)
        torch.nn.init.normal_(self.out.weight, 0., 0.001)
        torch.nn.init.normal_(self.out.bias, 0., 0.001)

    def forward(self, sents):
        max_len = max(len(sent) for sent in sents)
        nums = []
        for sent in sents:
            sent = ['<BOS>'] + sent + (max_len-len(sent)) * ['<PAD>']
            nums.append([self.vocab[w] for w in sent])
        nums = torch.tensor(nums)
        vecs = self.word_embedding[nums] + self.pos_encoding[:max_len+1,:]
        result = self.out(self.transformer(vecs.transpose(0,1))[0,:,:])
        return result

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

# For reference, train a PCFG and show loss.

pcfg = collections.defaultdict(dict)
for lhs in cfg:
    z = sum(cfg[lhs].values())
    for rhs in cfg[lhs]:
        pcfg[lhs][rhs] = cfg[lhs][rhs] / z
logp = 0.        
for _, tree in traindata:
    for node in tree.bottomup():
        if len(node.children) > 0:
            lhs = node.label
            rhs = tuple(child.label for child in node.children)
            logp += math.log(pcfg[lhs][rhs])
print(f'optimal_loss={-logp}')


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

fgg = fggs.FGG(fggs.EdgeLabel('TOP', [fggs.NodeLabel('batch')], is_nonterminal=True))

rules = {}
for lhs in cfg:
    for rhs in cfg[lhs]:
        hrhs = fggs.Graph()

        b = hrhs.new_node('batch')

        # One nonterminal edge for each nonterminal on the rhs of
        # the CFG rule
        for x in rhs:
            if isinstance(x, Nonterminal):
                hrhs.new_edge(x, [b], is_nonterminal=True)

        # One terminal edge for the weight of the CFG rule
        el = f'{repr(lhs)} -> {" ".join(map(repr, rhs))}'
        rules[lhs, rhs] = el # save for later use
        hrhs.new_edge(el, [b], is_terminal=True)

        hrhs.ext = [b]
        fgg.new_rule(lhs, hrhs)

fgg.new_finite_domain('batch', range(args.minibatch_size))
for el in rules.values():
    fgg.new_finite_factor(el, torch.zeros(args.minibatch_size))

### Factorize the FGG into smaller rules.

fgg = fggs.factorize_fgg(fgg)
encoder = Encoder(svocab, len(fgg.factors))

# It's helpful (though not essential) to initialize parameters so that
# Z is finite. To do this, we do a quick pre-training step using SGD
# with a fairly large setting for gradient clipping.

opt = torch.optim.SGD(encoder.parameters(), lr=1e-2)
for epoch in range(100):
    weights = [clip_grad_value(encoder.out.bias, 100.)]
    while len(weights) < args.minibatch_size:
        weights.append(torch.zeros(weights[0].size()))
    weights = torch.stack(weights, dim=0)
    for fi, fac in enumerate(fgg.factors.values()):
        fac.weights = weights[:,fi]
    z = fggs.sum_product(fgg, method='newton', tol=1e-3, kmax=10, semiring=fggs.LogSemiring(device=args.device))[:1]
    print(f'Z={z.item()}')
    if z < math.inf: break
    opt.zero_grad()
    z.backward()
    opt.step()
        
### Train the model.

def minibatches(iterable, size):
    b = []
    for i, x in enumerate(iterable):
        if i % size == 0 and len(b) > 0:
            yield b
            b = []
        b.append(x)
    if len(b) > 0:
        yield b

opt = torch.optim.Adam(encoder.parameters(), lr=3e-4)
for epoch in range(1000):
    random.shuffle(traindata)
    train_loss = 0.
    with tqdm.tqdm(total=len(traindata)) as progress:
        for minibatch in minibatches(traindata, args.minibatch_size):
            loss = 0.
            w = encoder([swords for swords, _ in minibatch])
            # Gradient clipping should happen right before the FGG
            weights = clip_grad_value(w, 10.) # bug: clips whole minibatch
            weights = torch.nn.functional.pad(weights, (0,0,0,args.minibatch_size-len(minibatch)))
            for fi, fac in enumerate(fgg.factors.values()):
                fac.weights = weights[:,fi]

            for si, (swords, ttree) in enumerate(minibatch):
                # Compute the weight of the tree.
                for node in ttree.bottomup():
                    if len(node.children) > 0:
                        lhs = node.label
                        rhs = tuple(child.label for child in node.children)
                        loss -= fgg.factors[rules[lhs, rhs]].weights[si]
                        
            # Compute the total weight of all trees.
            loss += fggs.sum_product(fgg, method='newton', tol=1e-3, kmax=10, semiring=fggs.LogSemiring(device=args.device))[:len(minibatch)].sum()
            
            train_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()
            progress.update(len(minibatch))
    print(f'epoch={epoch+1} train_loss={train_loss}')
