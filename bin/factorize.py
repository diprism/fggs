#!/usr/bin/env python3

import json
import sys
import argparse

from fggs import factorize, json_to_hrg, hrg_to_json

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Factorize an HRG.')
    ap.add_argument('input', metavar='json')
    ap.add_argument('-o', dest='output', metavar='json', required=True)
    ap.add_argument('-m', dest='method', metavar='method', default='min_fill', choices=['min_fill', 'quickbb', 'acb'])

    args = ap.parse_args()

    g_in = json_to_hrg(json.load(open(args.input)))
    g_out = factorize(g_in, method=args.method)

    width_in = max(len(r.rhs().nodes()) for r in g_in.all_rules())
    width_out = max(len(r.rhs().nodes()) for r in g_out.all_rules())
    print(f'maximum rule width: {width_in} -> {width_out}', file=sys.stderr)
    
    with open(args.output, 'w') as outfile:
        json.dump(hrg_to_json(g_out), outfile, indent=4)
        
