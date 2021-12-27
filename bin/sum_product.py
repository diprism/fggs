#!/usr/bin/env python3

import json
import sys
import argparse

import fggs

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Compute the sum-product of an FGG.')
    ap.add_argument('fgg', metavar='json')
    ap.add_argument('-m', metavar='method', dest='method', default='fixed-point', choices=['fixed-point', 'newton', 'linear', 'broyden'])
    ap.add_argument('-w', metavar=('factor', 'weights'), dest='weights', action='append', default=[], nargs=2)

    args = ap.parse_args()

    fgg = fggs.json_to_fgg(json.load(open(args.fgg)))

    for name, weights in args.weights:
        el = fgg.grammar.get_edge_label(name)
        weights = json.loads(weights)
        if el not in fgg.interp.factors:
            doms = [fgg.interp.domains[nl] for nl in el.type()]
            fgg.interp.add_factor(el, fggs.CategoricalFactor(doms, weights))
        else:
            fgg.interp.factors[el].weights = weights
    
    print(json.dumps(fggs.formats.weights_to_json(fggs.sum_product(fgg, method=args.method))))
