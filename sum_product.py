import warnings, torch

def _formatwarning(message, category, filename=None, lineno=None, file=None, line=None):
    return '%s:%s: %s: %s' % (filename, lineno, category.__name__, message)
warnings.formatwarning = _formatwarning

def fixed_point(F, psi_X0, tol=1e-8, maxiter=1000):
    k, psi_X1 = 0, F(psi_X0)
    while any(torch.max(torch.abs(psi_X0[X] - psi_X1[X])) > tol
            for X in psi_X0) and k <= maxiter:
        psi_X0, psi_X1 = psi_X1, F(psi_X1)
        k += 1
    if k > maxiter:
        warnings.warn('maximum iteration exceeded; convergence not guaranteed')
    return psi_X1

def sum_product(fgg, method='fixed-point'):
    def F(psi_X):
        psi_X = psi_X.copy()
        for nt_name in fgg._nonterminals:
            tau_R = []
            for rule in fgg.rules(nt_name):
                if len(rule.rhs().nodes()) > 52:
                    raise Exception('cannot assign an index to each node (maximum of 52 supported)')
                Xi_R = {id: chr((ord('a') if i < 26 else ord('A')) + i % 26)
                    for i, id in enumerate(rule.rhs()._node_ids)}
                indexing, tensors = [], []
                for edge in rule.rhs().edges():
                    indexing.append([Xi_R[node.id()] for node in edge.nodes()])
                    if edge.label().is_terminal():
                        tensors.append(edge.label().factor().weights())
                    else:
                        tensors.append(psi_X[edge.label().name()])
                for i, tensor in enumerate(tensors):
                    if not isinstance(tensor, torch.Tensor):
                        tensors[i] = torch.tensor(tensor, dtype=torch.float)
                    elif not tensor.is_floating_point():
                        tensors[i] = torch.float()
                indexing = ','.join([''.join(indices) for indices in indexing]) + '->'
                external = [Xi_R[node.id()] for node in rule.rhs().ext()]
                if external: indexing += ''.join(external)
                tau_R.append(torch.einsum(indexing, *tensors))
            psi_X[nt_name] = sum(tau_R) if len(tau_R) > 1 else tau_R[0]
        return psi_X
    psi_X = {}
    for nt_name in fgg._nonterminals:
        for _ in fgg.rules(nt_name):
            size = [node_label.domain().size() for node_label in fgg._nonterminals[nt_name]._node_labels]
            psi_X[nt_name] = torch.full(size, fill_value=0.0, dtype=torch.float)
    if method == 'fixed-point':
        return fixed_point(F, psi_X)[fgg.start_symbol().name()]
    else: raise ValueError('unsupported method for computing sum-product')
