#!/usr/bin/env python3

import json
import sys
import argparse
import torch
import fggs

def error(s):
    print('error:', s, file=sys.stderr)
    exit(1)

def string_to_tensor(s, name="tensor", shape=None):
    try:
        j = json.loads(s)
    except json.decoder.JSONDecodeError as e:
        error(f"couldn't understand {name}: {e}")
    t = torch.tensor(j, dtype=float)
    if shape is not None and t.shape != shape:
        error(f"{name} should have shape {shape}")
    return t

def tensor_to_string(t):
    return json.dumps(fggs.formats.weights_to_json(t))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Compute the sum-product of an FGG.')
    ap.add_argument('fgg', metavar='<fgg>', help='the FGG, in JSON format')
    ap.add_argument('-m', metavar='<method>', dest='method', default='newton', choices=['fixed-point', 'linear', 'newton'], help='use <method> (fixed-point, linear, or newton)')
    ap.add_argument('-w', metavar=('<factor>', '<weights>'), dest='weights', action='append', default=[], nargs=2, help="set <factor>'s weights to <weights>")
    ap.add_argument('-o', metavar='<out_weights>', dest='out_weights', help='for -g and -e options, weight the elements of sum-product by <weights> (default: all 1)')
    ap.add_argument('-g', dest='grad', action='store_true', help='compute gradient with respect to factors from -w option')
    ap.add_argument('-e', dest='expect', action='store_true', help='compute expected counts of factors from -w option')

    args = ap.parse_args()

    fgg = fggs.json_to_fgg(json.load(open(args.fgg)))

    extern_weights = {}
    for name, weights in args.weights:
        el = fgg.get_edge_label(name)
        weights = string_to_tensor(weights, f"<weights> for {name}", fgg.shape(el))
        if args.grad or args.expect:
            weights.requires_grad_()
        extern_weights[name] = weights
        if name not in fgg.factors:
            fgg.new_finite_factor(name, weights)
        else:
            fgg.factors[name].weights = weights

    if args.out_weights:
        out_weights = string_to_tensor(args.out_weights, f"<out_weights>", fgg.shape(fgg.start_symbol))
    else:
        out_weights = 1.

    for el in fgg.terminals():
        if el.name not in fgg.factors:
            error(f'factor {el.name} needs weights (use -w option)')
        fac = fgg.factors[el.name]
        fac.weights = torch.as_tensor(fac.weights, dtype=float)

    z = fggs.sum_product(fgg, method=args.method)
    print(json.dumps(fggs.formats.weights_to_json(z)))

    if args.grad or args.expect:
        if not args.weights:
            error('the -g and -e options require the -w option')
        f = (z * out_weights).sum()
        f.backward()
        for name, weights in extern_weights.items():
            grad = weights.grad
            
            if args.grad:
                print(f'grad[{name}]:', tensor_to_string(grad))

            if args.expect:
                expect = grad * weights / f
                print(f'E[#{name}]:', tensor_to_string(expect))
