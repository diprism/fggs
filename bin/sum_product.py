#!/usr/bin/env python3

import json
import sys
import argparse
import torch
import fggs

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Compute the sum-product of an FGG.')
    ap.add_argument('fgg', metavar='json')
    ap.add_argument('-m', metavar='method', dest='method', default='newton', choices=['fixed-point', 'newton', 'linear'])
    ap.add_argument('-w', metavar=('factor', 'weights'), dest='weights', action='append', default=[], nargs=2)

    args = ap.parse_args()

    fgg = fggs.json_to_fgg(json.load(open(args.fgg)))

    for name, weights in args.weights:
        weights = json.loads(weights)
        if name not in fgg.factors:
            fgg.new_factor(name, weights)
        else:
            fgg.factors[name].weights = weights

    for el, fac in fgg.factors.items():
        fac.weights = torch.as_tensor(fac.weights, dtype=torch.get_default_dtype())
    
    print(json.dumps(fggs.formats.weights_to_json(fggs.sum_product(fgg, method=args.method))))
