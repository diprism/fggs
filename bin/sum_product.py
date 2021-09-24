#!/usr/bin/env python3

import json
import sys
import argparse

from fggs import FGG, sum_product, json_to_hrg, json_to_interp

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Compute the sum-product of an FGG.')
    ap.add_argument('hrg', metavar='json')
    ap.add_argument('interp', metavar='json')
    ap.add_argument('-m', metavar='method', dest='method', default='fixed-point', choices=['fixed-point', 'broyden'])
    ap.add_argument('-w', metavar=('factor', 'weights'), dest='weights', action='append', default=[], nargs=2)

    args = ap.parse_args()

    fgg = FGG(json_to_hrg(json.load(open(args.hrg))),
              json_to_interp(json.load(open(args.interp))))

    for name, weights in args.weights:
        el = fgg.grammar.get_terminal(name)
        fgg.interp.factors[el]._weights = json.loads(weights)
    
    print(sum_product(fgg, method=args.method))
