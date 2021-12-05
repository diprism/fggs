# fggs: Factor Graph Grammars in Python
![workflow](https://github.com/diprism/fgg-implementation/actions/workflows/ci.yaml/badge.svg)

Factor graph grammars (FGGs) are hyperedge replacement graph grammars for factor graphs. They generate sets of factor graphs and can describe a more general class of models than plate notation and many other formalisms can. Moreover, inference can be done on FGGs without enumerating all the generated factor graphs.

This library implements FGGs in Python and is compatible with PyTorch.

FGGs are described in the following paper:

David Chiang and Darcey Riley. [Factor graph grammars.](https://arxiv.org/abs/2010.12048) In Proc. NeurIPS. 2020.

This code is written by David Chiang, Darcey Riley, and Kenneth Sible, at the University of Notre Dame, and is licensed under the MIT License.

## Building

To build the documentation, run `make docs` in the project root.
