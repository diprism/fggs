# fggs: Factor Graph Grammars in Python
![workflow](https://github.com/diprism/fgg-implementation/actions/workflows/ci.yaml/badge.svg)

Factor graph grammars (FGGs) are hyperedge replacement graph grammars for factor graphs. They generate sets of factor graphs and can describe a more general class of models than plate notation and many other formalisms can. Moreover, inference can be done on FGGs without enumerating all the generated factor graphs.

This library implements FGGs in Python and is compatible with PyTorch (tested with Python >= 3.7 and PyTorch >= 3.8).

## Building

To build the documentation, run `make docs` in the project root.

## Using

See `examples/parser/parser.py` for an example of using the package to
train a simple FGG.
