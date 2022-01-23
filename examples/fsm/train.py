import torch
import fggs
import collections
import tqdm
import os
import sys
import subprocess
import argparse
import random
import json
import copy
from fggs import factorize, json_to_hrg, json_to_fgg, hrg_to_json

DEFAULT_SEED = 567
DEFAULT_VOCAB_SIZE = 100
DEFAULT_NUM_STATES = 100
DEFAULT_NUM_EPOCHS = 100
COMPILER_EXE="./compiler.exe"
UNK = '_UNK_'

def parse_args():
    ap = argparse.ArgumentParser()
    #ap.add_argument('-train-src', dest="train_src", help="Train source file")
    #ap.add_argument('-dev-src', dest="dev_src", help="Dev source file")
    #ap.add_argument('-test-src', dest="test_src", help="Test source file")
    ap.add_argument('-compiler', dest='compiler', help="Location of compiler executable")
    ap.add_argument('-train-tgt', dest="train_tgt", help="Train target file")
    ap.add_argument('-dev-tgt', dest="dev_tgt", help="Dev target file")
    ap.add_argument('-test-tgt', dest="test_tgt", help="Test target file")
    ap.add_argument('-seed', dest="seed", default=DEFAULT_SEED, type=int, help="Random seed")
    #ap.add_argument('-src-size', dest="src_size", default=DEFAULT_VOCAB_SIZE, type=int, help="Source vocabulary size")
    ap.add_argument('-tgt-size', dest="tgt_size", default=DEFAULT_VOCAB_SIZE, type=int, help="Target vocabulary size")
    #ap.add_argument('-num-src-states', dest="num_src_states", default=DEFAULT_NUM_STATES, type=int, help="Number of source FSM states")
    ap.add_argument('-num-tgt-states', dest="num_tgt_states", default=DEFAULT_NUM_STATES, type=int, help="Number of target FSM states")
    ap.add_argument('-num-epochs', dest="num_epochs", default=DEFAULT_NUM_EPOCHS, type=int, help="Number of epochs")
    return ap.parse_args()

def read_file(fp):
    with open(fp) as fh:
        #return [line.strip().split() for line in fh]
        return [line.strip() for line in fh]

def make_vocab(data, k):
    # First, create a dict mapping words to their frequency
    vocab = {UNK: float('inf')}
    for line in data:
        for word in line:
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1

    # Now trim down to top-k most frequent words
    sorted_vocab = sorted(vocab.items(), key=(lambda wo: wo[1]), reverse=True)[:k]
    word_to_tok = {w: i for i, (w, _) in enumerate(sorted_vocab)}
    tok_to_word = [w for w, _ in sorted_vocab]
    return word_to_tok, tok_to_word

def tokenize(data, vocab):
    for line in data:
        yield [vocab[word] if word in vocab else vocab[UNK]]

class Dataset:
    def __init__(self, train_file, dev_file, test_file, vocab_size, batch_size = None):
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.vocab_size = vocab_size
        self.train_data_raw = read_file(train_file)
        self.dev_data_raw = read_file(dev_file)
        self.test_data_raw = read_file(test_file)
        #self.train_length = len(self.train_data_raw)
        #self.dev_length = len(self.dev_data_raw)
        #self.test_length = len(self.test_data_raw)
        self.word_to_tok, self.tok_to_word = make_vocab(self.train_data_raw, vocab_size)
        self.train_data = [self.forward_translate(line) for line in self.train_data_raw]
        self.dev_data = [self.forward_translate(line) for line in self.dev_data_raw]
        self.test_data = [self.forward_translate(line) for line in self.test_data_raw]
        self.batch_size = batch_size

    def forward_translate(self, line):
        return [self.word_to_tok[word if word in self.word_to_tok else UNK] for word in line]

    def back_translate(self, line):
        return [self.tok_to_word[tok] for tok in line]

    def batches(self, data):
        if self.batch_size is None: # Yield one line at a time
            yield from self.train_data
        else: # Yield batches with self.batch_size lines
            rem = len(data)
            itr = iter(data)
            while rem > 0:
                rem -= self.batch_size
                yield [next(itr) for _ in range(min(self.batch_size, rem))]
    
    def train_batches(self):
        yield from self.batches(self.train_data)

    def dev_batches(self):
        yield from self.batches(self.dev_data)

    def test_batches(self):
        yield from self.batches(self.test_data)

def init_param_tensor(shape):
    "Instantiates a random probabilistic tensor t such that sum(exp(t)) = 1"
    # First, instantiate from the uniform distribution
    t = torch.rand(shape)
    # Then, make all the probabilities sum to 1
    t = t/torch.sum(t)
    # Finally, take the log so that when we exp it later,
    # the probabilities still sum to 1
    t = torch.log(t)
    # Now return the tensor, with gradient
    return t.clone().detach().requires_grad_(True)

def communicate_fgg(input_str, compiler):
    "Open a process running the compiler, send it input_str, and returned the output"
    p = subprocess.Popen([compiler],
                         stdout = subprocess.PIPE,
                         stdin = subprocess.PIPE,
                         stderr = subprocess.PIPE)
    out, err = p.communicate(input=str.encode(input_str))
    p.kill()
    if err:
        print(err, file=sys.stderr)
        exit(1)
    else:
        return json.loads(out)

FGG_STR_SPECIFIC = '''
data String = Cons Sigma String | Nil;

data Q = {};
data Sigma = {};

data Maybe = Some Q Sigma | None;
data Unit = unit;

extern delta : Maybe -> Maybe;

define gen2 : String -> Maybe -> Unit =
  \ s : String. \ m : Maybe. case delta m of
    | None -> (case s of Nil -> unit | Cons _ _ -> sample fail : Unit)
    | Some q c -> (case s of Nil -> sample fail : Unit | Cons c' s' -> (if (c == c') then gen2 s' (Some q c) else (sample fail : Unit)));

gen2 ({}) None;
'''

FGG_STR_TOTAL = '''
data Q = {};
data Sigma = {};

data Maybe = Some Q Sigma | None;
data Unit = unit;

extern delta : Maybe -> Maybe;

define gen : Maybe -> Unit =
  \ m : Maybe. case delta m of None -> unit | Some q c -> gen (Some q c);

gen None;
'''



def get_fgg_str(num_states, vocab_size, line=None):
    states = ' | '.join(f'q{n}' for n in range(num_states))
    sigma = ' | '.join(f'w{n}' for n in range(vocab_size))
    if line:
        string = 'Nil'
        for word in reversed(line):
            string = f'Cons w{word} ({string})'
        return FGG_STR_SPECIFIC.format(states, sigma, string)
    else:
        return FGG_STR_TOTAL.format(states, sigma)

def convert_json_to_fgg(fgg_json, delta_weights, num_states, vocab_size):
    fgg_ = fggs.json_to_fgg(fgg_json)
    grammar, interp = fgg_.grammar, fgg_.interp

    maybe_dom = fggs.FiniteDomain(set(f'mdom{i}' for i in range(num_states * vocab_size + 1))) # Doesn't really matter what these are named, right?
    delta_fac = fggs.CategoricalFactor([maybe_dom, maybe_dom], torch.exp(delta_weights))
    maybe_nl = fggs.NodeLabel('Maybe')
    el = fggs.EdgeLabel('0', [maybe_nl, maybe_nl], is_terminal=True)
    interp.factors[el] = delta_fac
    return fggs.FGG(grammar, interp)

def get_fgg(input_str, delta_weights, num_states, vocab_size, compiler):
    fgg_json = communicate_fgg(get_fgg_str(num_states, vocab_size, input_str), compiler)
    fgg_json_total = communicate_fgg(get_fgg_str(num_states, vocab_size), compiler)
    return (convert_json_to_fgg(fgg_json, delta_weights, num_states, vocab_size),
            convert_json_to_fgg(fgg_json_total, delta_weights, num_states, vocab_size))

def clip_grad_nans(parameters):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in filter(lambda p: p.grad is not None, parameters):
        p.grad.data[torch.isnan(p.grad.data)] = 0.0

if __name__ == '__main__':
    args = parse_args()
    
    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    tgt_data = Dataset(args.train_tgt, args.dev_tgt, args.test_tgt, args.tgt_size)
    maybe_dom_size = args.num_tgt_states * args.tgt_size + 1
    tgt_delta_params = init_param_tensor([maybe_dom_size, maybe_dom_size])
    
    print('begin training')
    opt = torch.optim.Adam([tgt_delta_params], lr=1e-1)
    
    for epoch in range(args.num_epochs):
        train_loss = 0.0
        with tqdm.tqdm(total=len(tgt_data.train_data)) as progress:
            #for src_line, tgt_line in zip(src_data.train_batches(), tgt_data.train_batches()):
            for tgt_line in tgt_data.train_batches():
                specific_fgg, total_fgg = get_fgg(tgt_line, tgt_delta_params, args.num_tgt_states, args.tgt_size, args.compiler)
                # Newton's method currently works better than fixed-point iteration
                # for avoiding z = 0.
                z1 = fggs.sum_product(specific_fgg, method='newton', kmax=100, tol=1e-30)
                z2 = fggs.sum_product(total_fgg, method='newton', kmax=100, tol=1e-30)
                z = z1/z2
    
                loss = torch.log(z)
                train_loss += loss.item()
                
                print(loss.item())
    
                opt.zero_grad()
                loss.backward()
                clip_grad_nans(tgt_delta_params)
                opt.step()
    
                progress.update(1)
        print(tgt_delta_params)
        print(f'epoch={epoch+1} train_loss={train_loss}')
