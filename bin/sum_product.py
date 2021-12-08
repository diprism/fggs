#!/usr/bin/env python3

import json
import sys
import argparse

from fggs import FGG, sum_product, json_to_fgg

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Compute the sum-product of an FGG.')
    ap.add_argument('fgg', metavar='json')
    ap.add_argument('-m', metavar='method', dest='method', default='fixed-point', choices=['fixed-point', 'broyden'])
    ap.add_argument('-w', metavar=('factor', 'weights'), dest='weights', action='append', default=[], nargs=2)

    args = ap.parse_args()

    fgg = json_to_fgg(json.load(open(args.fgg)))

    for name, weights in args.weights:
        el = fgg.grammar.get_terminal(name)
        fgg.interp.factors[el].weights = json.loads(weights)
    
    print(sum_product(fgg, method=args.method))
