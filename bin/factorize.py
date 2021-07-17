#!/usr/bin/env python3

import json
import sys
import argparse

from fggs import factorize, json_to_fgg, fgg_to_json

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Compute the sum-product of an FGG.')
    ap.add_argument('input', metavar='json')
    ap.add_argument('-o', dest='output', metavar='json', required=True)
    ap.add_argument('-m', dest='method', metavar='method', default='min_fill', choices=['min_fill', 'quickbb', 'acb'])

    args = ap.parse_args()

    fgg_in = json_to_fgg(json.load(open(args.input)))
    fgg_out = factorize(fgg_in, method=args.method)

    width_in = max(len(r.rhs().nodes()) for r in fgg_in.all_rules())
    width_out = max(len(r.rhs().nodes()) for r in fgg_out.all_rules())
    print(f'maximum rule width: {width_in} -> {width_out}', file=sys.stderr)
    
    with open(args.output, 'w') as outfile:
        json.dump(fgg_to_json(fgg_out), outfile, indent=4)
        
