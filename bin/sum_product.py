#!/usr/bin/env python3

import json
import sys
import argparse

from fggs import sum_product, json_to_fgg

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Compute the sum-product of an FGG.')
    ap.add_argument('fgg', metavar='json')

    args = ap.parse_args()

    fgg = json_to_fgg(json.load(open(args.fgg)))
    
    print(sum_product(fgg))