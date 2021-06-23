from functools import reduce
import torch

MAX_ITER = 1000

def fixed_point(F, psi_X0, tol=1e-8):
    def flatten(iterable):
        for element in iterable:
            if isinstance(element, list):
                yield from flatten(element)
            else: yield element
    k, psi_X1 = 0, F(psi_X0)
    while any(flatten((torch.abs(psi_X0[i] - psi_X1[i]) > tol).tolist()
            for i in psi_X0)) and k <= MAX_ITER:
        psi_X0, psi_X1 = psi_X1, F(psi_X1)
        k += 1
    return psi_X1

def sum_product(fgg, method='fixed-point'):
    def F(psi_X, nonterminal=None):
        psi_X = psi_X.copy()
        nonterminals = [nonterminal] if nonterminal else (nonterminal.name()
            for nonterminal in fgg.nonterminals() if nonterminal != fgg.start_symbol())
        for nonterminal in nonterminals:
            tau_R = list()
            for rule in fgg.rules(nonterminal):
                Xi_R = {id: chr(ord('a') + i) for i, id in enumerate(rule.rhs()._node_ids)}
                indexing, tensors = [], []
                for edge in rule.rhs().edges():
                    indexing.append([Xi_R[node.id()] for node in edge.nodes()])
                    if edge.label().is_terminal():
                        tensors.append(edge.label().factor().weights())
                    else:
                        tensors.append(psi_X[edge.label().name()])
                for i, tensor in enumerate(tensors):
                    if not isinstance(tensor, torch.Tensor):
                        tensors[i] = torch.tensor(tensor, dtype=torch.double)
                    elif not tensor.is_floating_point():
                        tensors[i] = torch.double()
                indexing = ','.join([''.join(indices) for indices in indexing]) + '->'
                external = [Xi_R[node.id()] for node in rule.rhs().ext()]
                if external: indexing += ''.join(external)
                tau_R.append(torch.einsum(indexing, *tensors))
            psi_X[nonterminal] = reduce(lambda x, y: x + y, tau_R) if len(tau_R) > 1 else tau_R[0]
        return psi_X
    psi_X = dict()
    for name, nonterminal in ((nonterminal.name(), nonterminal) for nonterminal
            in fgg.nonterminals() if nonterminal != fgg.start_symbol()):
        for _ in fgg.rules(name):
            size = [node_label.domain().size() for node_label in nonterminal._node_labels]
            psi_X[name] = torch.full(size, fill_value=0.1, dtype=torch.double)
    if method == 'fixed-point':
        psi_X, start = fixed_point(F, psi_X), fgg.start_symbol().name()
        return F(psi_X, start)[start]
    else: raise ValueError('unsupported method for computing sum-product')
