import torch
import trees
import collections
import random
import argparse
import tqdm # type: ignore

ap = argparse.ArgumentParser()
ap.add_argument('trainfile')
ap.add_argument('--device', dest="device", default="cpu")
ap.add_argument('-b', dest="minibatch_size", type=int, default=1)
args = ap.parse_args()

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
    def __init__(self, vocab, cfg):
        super().__init__()
        self.vocab = vocab
        self.cfg = cfg
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
        self.out = torch.nn.Linear(dim, sum(len(cfg[lhs]) for lhs in cfg))
        torch.nn.init.normal_(self.out.weight, 0., 0.001)
        torch.nn.init.normal_(self.out.bias, 0., 0.001)

    def forward(self, words):
        words = ['<BOS>'] + words
        words = [self.vocab[w] for w in words]
        words = self.word_embedding[words] + self.pos_encoding[:len(words),:]
        logits = self.out(self.transformer(words.unsqueeze(1))[0,0,:])
        pcfg = {}
        i = 0
        for lhs in self.cfg:
            j = i + len(self.cfg[lhs])
            pcfg[lhs] = dict(zip(self.cfg[lhs], torch.log_softmax(logits[i:j], dim=0)))
            i = j
        assert j == len(logits)
        return pcfg

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

def minibatches(iterable, size):
    b = []
    for i, x in enumerate(iterable):
        if i % size == 0 and len(b) > 0:
            yield b
            b = []
        b.append(x)
    if len(b) > 0:
        yield b

encoder = Encoder(svocab, cfg)        
opt = torch.optim.Adam(encoder.parameters(), lr=3e-4)
for epoch in range(100):
    random.shuffle(traindata)
    train_loss = 0.
    with tqdm.tqdm(total=len(traindata)) as progress:
        for minibatch in minibatches(traindata, args.minibatch_size):
            loss = 0.
            for swords, ttree in minibatch:
                pcfg = encoder(swords)
                # Compute the weight of the tree.
                for node in ttree.bottomup():
                    if len(node.children) > 0:
                        lhs = node.label
                        rhs = tuple(child.label for child in node.children)
                        loss -= pcfg[lhs][rhs]
            train_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()
            
            progress.update(len(minibatch))
    print(f'epoch={epoch+1} train_loss={train_loss}')
