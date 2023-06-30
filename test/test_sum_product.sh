#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH

echo -n "simplefgg.json: "

ANS=`${PYTHON:-python} bin/sum_product.py test/simplefgg.json -w fac1 0.8 -w fac2 0.2 -B 2`
PASS=`echo "($ANS - 0.25) ^ 2 < 0.0000001" | bc`
if [ $PASS -eq 1 ]; then
    echo "passed"
else
    echo "failed"
    exit 1
fi
