#!/usr/bin/env -S python3 -OO

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
    t = fggs.json_to_weights(j)
    if shape is not None and t.shape != shape:
        error(f"{name} should have shape {shape}")
    return t

def tensor_to_dict_string(is_pretty, fgg, node, t):
    if is_pretty:
        return json.dumps(fggs.formats.weights_to_dict_json(fgg, node, t))
    else:
        return json.dumps(t.tolist())

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Compute the sum-product of an FGG.')
    ap.add_argument('fgg', metavar='<fgg>', help='the FGG, in JSON format')
    ap.add_argument('-m', metavar='<method>', dest='method', default='newton', choices=['fixed-point', 'linear', 'newton'], help='use <method> (fixed-point, linear, or newton)')
    ap.add_argument('-l', metavar='<tol>', dest='tol', type=float, default=1e-5, help="stop iterating when change is below <tol>")
    ap.add_argument('-k', metavar='<kmax>', dest='kmax', type=int, default=1000, help="iterate at most <kmax> times")
    ap.add_argument('-w', metavar=('<factor>', '<weights>'), dest='weights', action='append', default=[], nargs=2, help="set <factor>'s weights to <weights>")
    ap.add_argument('-n', metavar=('<factor>', '<dim>'), dest='normalize', action='append', default=[], nargs=2, help="normalize <factor> along <dim>")
    ap.add_argument('-o', metavar='<out_weights>', dest='out_weights', help='for -g and -e options, weight the elements of sum-product by <weights> (default: all 1)')
    ap.add_argument('-g', dest='grad', action='store_true', help='compute gradient with respect to factor(s) from -w option(s)')
    ap.add_argument('-G', dest='grad_all', action='store_true', help='compute gradient with respect to all factors')
    ap.add_argument('-e', dest='expect', action='store_true', help='compute expected counts of factor(s) from -w option(s)')
    ap.add_argument('-t', dest='trace', action='store_true', help='print out all intermediate sum-products')
    ap.add_argument('-d', dest='double', action='store_true', help='use double-precision floating-point')
    ap.add_argument('-p', dest='pretty', action='store_true', help='pretty print weights together with the value names')
    ap.add_argument('-j', dest='j_precompute', action='store_true', help='precompute products while solving the Jacobian (faster for rules with many edges)')
    ap.add_argument('--tikz', metavar='<file>', dest='tikz', default=None, help='convert the input JSON to tikz and write the output to the given file')

    args = ap.parse_args()
    if args.double: torch.set_default_dtype(torch.float64)
    if (args.grad or args.expect) and not args.weights:
        error('the -g and -e options require the -w option')

    fgg = fggs.json_to_fgg(json.load(open(args.fgg)))

    extern_weights = {}
    for name, weights in args.weights:
        if not fgg.has_edge_label_name(name):
            error(f'FGG does not have an edge label named {name}')
        el = fgg.get_edge_label(name)

        weights = string_to_tensor(weights, f"weights for {name}", fgg.shape(el))
        if args.grad or args.expect:
            weights.requires_grad_()
        extern_weights[name] = weights

        # el can either be a terminal or a nonterminal.
        # If it's a nonterminal, create a rule for it that rewrites to a factor.
            
        if el.is_nonterminal:
            if len(fgg.rules(el)) > 0:
                error(f'FGG already has rules for nonterminal {name}')
            weights_name = name + "_weights"
            assert not fgg.has_edge_label_name(weights_name)
            rhs = fggs.Graph()
            nodes = [fggs.Node(nl) for nl in el.type]
            rhs.new_edge(weights_name, nodes, is_terminal=True)
            rhs.ext = nodes
            fgg.new_rule(name, rhs)
            name = weights_name

        if name in fgg.factors:
            error(f'FGG already has a factor named {name}')
        fgg.new_finite_factor(name, weights)

    for name, dim in args.normalize:
        if name not in fgg.factors:
            weights_name = name + "_weights"
            if weights_name in fgg.factors:
                name = weights_name
            else:
                error(f'cannot normalize nonexistent factor {name}')
        fgg.factors[name].weights = torch.nn.functional.normalize(
            fgg.factors[name].weights, p=1, dim=int(dim))

    if args.tikz:
        with open(args.tikz, "w") as f:
            tex = fggs.hrg_to_tikz(fgg)
            f.write(tex)
        sys.exit(0)

    if args.out_weights:
        out_weights = string_to_tensor(args.out_weights, f"<out_weights>", fgg.shape(fgg.start))
    else:
        out_weights = 1.

    for el in fgg.terminals():
        if el.name not in fgg.factors:
            error(f'factor {el.name} needs weights (use -w option)')

    if args.grad_all:
        for w in fgg.factors.values():
            w.weights.requires_grad_()

    zs = fggs.sum_products(fgg, method=args.method, tol=args.tol, kmax=args.kmax, j_precompute=args.j_precompute)
    z = zs[fgg.start]

    if args.trace:
        for el, zel in zs.items():
            print(tensor_to_dict_string(args.pretty, fgg, el, zel))
    else:
        print(tensor_to_dict_string(args.pretty, fgg, fgg.start, z))

    if (args.grad_all or args.grad or args.expect) and len(fgg.factors) > 0:
        f = (z * out_weights).sum()
        f.backward()

        if args.grad_all:
            grad_weights = {k: v.weights for k, v in fgg.factors.items()}
        else:
            grad_weights = extern_weights

        for name, weights in grad_weights.items():
            edge = fgg.get_edge_label(name)
            grad = weights.grad
            
            if args.grad or args.grad_all:
                print(f'grad[{name}]:', tensor_to_dict_string(args.pretty, fgg, edge, grad))

            if args.expect:
                expect = grad * weights / f
                print(f'E[#{name}]:', tensor_to_dict_string(args.pretty, fgg, edge, expect))
