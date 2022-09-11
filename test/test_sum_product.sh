#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH

echo -n "simplefgg.json: "

ANS=`python bin/sum_product.py test/simplefgg.json -w X1 0.8 -w X2 0.2`
PASS=`echo "($ANS - 0.25) ^ 2 < 0.0000001" | bc`
if [ $PASS -eq 1 ]; then
    echo "passed"
else
    echo "failed"
    exit 1
fi
