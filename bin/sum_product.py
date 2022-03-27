#!/usr/bin/env python3

import json
import sys
import argparse
import torch
import fggs

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Compute the sum-product of an FGG.')
    ap.add_argument('fgg', metavar='json')
    ap.add_argument('-m', metavar='method', dest='method', default='fixed-point', choices=['fixed-point', 'newton', 'linear'])
    ap.add_argument('-w', metavar=('factor', 'weights'), dest='weights', action='append', default=[], nargs=2)

    args = ap.parse_args()

    fgg = fggs.json_to_fgg(json.load(open(args.fgg)))

    for name, weights in args.weights:
        weights = torch.tensor(json.loads(weights), dtype=torch.get_default_dtype())
        if name not in fgg.interp.factors:
            fgg.new_factor(name, weights)
        else:
            fgg.interp.factors[name].weights = weights
    
    print(json.dumps(fggs.formats.weights_to_json(fggs.sum_product(fgg, method=args.method))))
