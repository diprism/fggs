#!/usr/bin/env python3

from sum_product import sum_product
from formats import json_to_fgg
import json
import sys

def infer_fgg(fgg, weights, method='fixed-point'):
    for name, weights in factors:
        fgg.get_terminal(name).factor._weights = weights
    return sum_product(fgg, method=method)

if __name__ == '__main__':
    try:
        program, factors_args = sys.argv[0], sys.argv[1:]
        factors = []
        while factors_args and factors_args[0] in ['--factor', '-f']:
            factors.append(factors_args[1], exec(factors_args[2]))
            factors_args = factors_args[3:]
    except:
        print(f"Usage: {sys.argv[0]} --factor p [0.0, 0.5, 0.3, 0.2] --factor... < FILE.json", file=sys.stderr)
        exit(1)
    
    fgg = json_to_fgg(json.load(sys.stdin))
    print(infer_fgg(fgg, weights))
