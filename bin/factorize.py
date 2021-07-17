#!/usr/bin/env python3

import json
import sys
import argparse

from fggs import factorize, json_to_fgg, fgg_to_json

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Compute the sum-product of an FGG.')
    ap.add_argument('input', metavar='json')
    ap.add_argument('-o', dest='output', metavar='json', required=True)

    args = ap.parse_args()

    fgg = json_to_fgg(json.load(open(args.input)))
    fgg = factorize(fgg)
    with open(args.output, 'w') as outfile:
        json.dump(fgg_to_json(fgg), outfile, indent=4)
        
