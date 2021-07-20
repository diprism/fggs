# Example 19 from FGG paper

# PCFG should be in Chomsky normal form in the format:
# X -> Y Z # count
# The first rule's LHS is the start symbol.

import sys
import re
import collections
import fggs
import json

cfg = collections.defaultdict(dict)
nonterminals = set()
terminals = set()
start = None
for line in open(sys.argv[1]):
    m = re.fullmatch(r'(\S+)\s*->\s*(.*)\s*#\s*(\S+)\s*', line)
    lhs = m.group(1)
    if start is None: start = lhs
    rhs = tuple(m.group(2).split())
    if len(rhs) == 2:
        nonterminals.update(rhs)
    elif len(rhs) == 1:
        terminals.update(rhs)
    else:
        raise ValueError()
    count = int(m.group(3))
    cfg[lhs][rhs] = count
for lhs in cfg:
    z = sum(cfg[lhs].values())
    for rhs in cfg[lhs]:
        cfg[lhs][rhs] /= z
nonterminals = list(nonterminals)
nonterminal_index = {x:i for i,x in enumerate(nonterminals)}
terminals = list(terminals)
terminal_index = {x:i for i,x in enumerate(terminals)}

g = fggs.json_to_fgg(json.load(open('pcfg.json')))

# Replace domain of nonterminals. This is messy, but we can't
# change g.get_node_label('N').domain because it's immutable.
dom = g.get_node_label('N').domain
dom._values = nonterminals
dom._value_index = nonterminal_index

# Similarly for terminals
dom = g.get_node_label('W').domain
dom._values = terminals
dom._value_index = terminal_index

# Similarly for factors
fac = g.get_edge_label('start').factor
fac._weights = [float(x == start) for x in nonterminals]

fac = g.get_edge_label('binary').factor
fac._weights = [
    [
        [cfg[x].get((y,z), 0.) for z in nonterminals] for y in nonterminals
    ]
    for x in nonterminals
]

fac = g.get_edge_label('terminal').factor
fac._weights = [
    [cfg[x].get((y,), 0.) for y in terminals]
    for x in nonterminals
]

with open(sys.argv[2], 'w') as outfile:
    json.dump(fggs.fgg_to_json(g), outfile, indent=4)
    
