#!/usr/bin/env bash

# Run from the fgg-implementation directory (not fgg-implementation/examples/fsm)

python3 examples/fsm/train.py --train-tgt examples/fsm/train_data_short.txt --dev-tgt examples/fsm/train_data_short.txt --test-tgt examples/fsm/train_data_short.txt --seed 15 --tgt-size 2 --num-tgt-states 1 --num-epochs 1 --compiler ../compiler/compiler.exe 
