__all__ = ['viterbi']

from fggs.fggs import FGG, FactorGraph, HRGRule, Node, Edge, EdgeLabel
from fggs.semirings import Semiring, ViterbiSemiring
from fggs.utils import nonterminal_graph, scc
from fggs.derivations import *
from fggs.sum_product import FGGMultiShape
from fggs.multi import MultiTensor

from typing import Dict, Tuple, NamedTuple, Set, List, Iterable, Optional
import torch
from torch import Tensor
import torch_semiring_einsum

# The Viterbi algorithm can be thought of as a sum-product in a
# special semiring, but there are enough little differences that it
# seems easier to implement it separately.

def sum_product_edges(fgg: FGG, rule: HRGRule, *inputses: MultiTensor, semiring: Semiring) -> Tuple[Tensor, Optional[Tensor]]:

    connected: Set[Node] = set()
    indexing: List[Iterable[Node]] = []
    tensors: List[Tensor] = []

    for edge in rule.rhs.edges():
        connected.update(edge.nodes)
        indexing.append(edge.nodes)
        for inputs in reversed(inputses):
            if edge.label in inputs:
                tensors.append(inputs[edge.label])
                break
        else:
            # One argument to einsum will be the zero tensor, so just return zero
            return (semiring.zeros(fgg.shape(rule.rhs.ext)), None)

    ptr: torch.Tensor
    if len(indexing) > 0:
        # Each node corresponds to an index, so choose a letter for each
        if len(connected) > 26:
            raise Exception('cannot assign an index to each node')
        node_to_index = {node: chr(ord('a') + i) for i, node in enumerate(connected)}

        equation = ','.join([''.join(node_to_index[n] for n in indices) for indices in indexing]) + '->'

        # If an external node has no edges, einsum will complain, so remove it.
        equation += ''.join(node_to_index[node] for node in rule.rhs.ext if node in connected)

        compiled = torch_semiring_einsum.compile_equation(equation)
        out, ptr = torch_semiring_einsum.log_viterbi_einsum_forward(compiled, *tensors, block_size=1)
    else:
        out = semiring.from_int(1)
        ptr = torch.empty(size=[0], dtype=torch.int, device=semiring.device)

    # Restore any external nodes that were removed.
    if out.ndim < len(rule.rhs.ext):
        eshape = fgg.shape(rule.rhs.ext)
        vshape = [s if n in connected else 1 for n, s in zip(rule.rhs.ext, eshape)]
        rshape = [1 if n in connected else s for n, s in zip(rule.rhs.ext, eshape)]
        out = out.view(*vshape).repeat(*rshape)
        nint = ptr.shape[-1]
        ptr = ptr.view(*vshape, nint).repeat(*rshape, nint)

    return out, ptr

def F_viterbi(fgg: FGG, x: MultiTensor, inputs: MultiTensor, semiring: Semiring):
    """
    Store the back-pointers as (maximum[nt], lhs_pointer[nt], rhs_pointer[nt]) where
      maximum[nt][ext_asst] is the maximum weight of ext_asst
      lhs_pointer[nt][ext_asst] is the index of the rule in the best derivation
      rhs_pointer[nt][ri][ext_asst] is the best assignment to the internal nodes of rule ri
    """
    Fx = MultiTensor(x.shapes, x.semiring)
    lhs_pointer = {n:torch.zeros(fgg.shape(n), dtype=torch.int, device=semiring.device) for n in x.shapes[0]}
    rhs_pointer: Dict = {n:[] for n in x.shapes[0]}
    for n in x.shapes[0]:
        for ri, rule in enumerate(fgg.rules(n)):
            tau_rule, pointer = sum_product_edges(fgg, rule, x, inputs, semiring=semiring)
            if n in Fx:
                lhs_pointer[n].masked_fill_(tau_rule > Fx[n], ri)
                Fx[n] = torch.maximum(Fx[n], tau_rule)
            else:
                lhs_pointer[n].fill_(ri)
                Fx[n] = tau_rule
            rhs_pointer[n].append(pointer)

    return (Fx, lhs_pointer, rhs_pointer)

def viterbi(fgg: FGG, start_asst: Tuple[int,...], **opts) -> FGGDerivation:
    """Find the highest-weight derivation of an FGG.

    Parameters:
    - fgg: the FGG
    - start_asst: tuple of assignments to the start nonterminal symbol
    - opts: see documentation for `sum_product`

    Returns:
    - An FGGDerivation object representing the highest-weight derivation.
    """
    if len(start_asst) != fgg.start.arity:
        raise ValueError(f"Start assignment ({start_asst}) does not have same type as FGG's start symbol ({fgg.start.type})")
    
    semiring = opts.get('semiring', ViterbiSemiring())
    kmax = opts.get('kmax', 1000)
    tol = opts.get('tol', 1e-6)

    maximum: MultiTensor = {t:fgg.factors[t.name].weights for t in fgg.terminals()} # type: ignore
    lhs_pointer = {}
    rhs_pointer = {}
    for comp in scc(nonterminal_graph(fgg)):
        x = MultiTensor(FGGMultiShape(fgg, comp), semiring)
        
        # Since we only use fixed-point iteration, we only need two cases.
        trivial = False
        if len(comp) == 1:
            [nt] = comp
            if not any(e.label == nt for r in fgg.rules(nt) for e in r.rhs.edges()):
                trivial = True

        if trivial:
            # The component has one nonterminal and is acyclic
            x1, lp1, rp1 = F_viterbi(fgg, x, maximum, semiring)
        else:
            # General case: fixed-point iteration
            for k in range(kmax):
                x1, lp1, rp1 = F_viterbi(fgg, x, maximum, semiring)
                if x.allclose(x1, tol): break
                x = x1
            
        maximum.update(x1)
        lhs_pointer.update(lp1)
        rhs_pointer.update(rp1)
    
    # Reconstruct derivation

    def reconstruct(nt: EdgeLabel, nt_asst: Tuple[int,...]) -> FGGDerivation:
        # lhs_pointer[nt][nt_asst] is the index of the first
        # rule used in the best derivation starting with edge.
        ri = lhs_pointer[nt][nt_asst]
        rule = list(fgg.rules(nt))[ri]

        # rhs_pointer[nt][ri][nt_asst] is the best assignment to the
        # internal nodes of rule.rhs. The nodes are listed in order of
        # appearance when iterating over the edges of rule.rhs.
        rhs_asst = dict(zip(rule.rhs.ext, nt_asst))
        ii = 0
        for e in rule.rhs.edges():
            for v in e.nodes:
                if v not in rhs_asst:
                    rhs_asst[v] = rhs_pointer[nt][ri][nt_asst][ii]
                    ii += 1
        assert ii == len(rhs_pointer[nt][ri][nt_asst])

        # Recurse on rhs nonterminal edges.
        child_derivs: Dict[Edge, FGGDerivation] = {}
        for e in rule.rhs.edges():
            if e.label.is_nonterminal:
                e_asst = tuple(rhs_asst[v] for v in e.nodes)
                child_derivs[e] = reconstruct(e.label, e_asst)
        
        return FGGDerivation(fgg, rule, rhs_asst, child_derivs)

    return reconstruct(fgg.start, start_asst)

