from fgg_representation import *

def json_to_fgg(j):
    """Convert an object loaded by json.load to an FGG."""
    import domain
    import torch
    g = FGGRepresentation()
    
    domains = {}
    for name, d in j['domains'].items():
        if d['class'] == 'finite':
            domains[name] = domain.FiniteDomain(name, d['values'])
        else:
            raise ValueError(f'invalid domain class: {d["type"]}')
        g.add_node_label(NodeLabel(name, domains[name]))

    for name, d in j['factors'].items():
        if d['function'] == 'categorical':
            size = [len(domains[l].values()) for l in d['type']]
            param = torch.empty(size, requires_grad=True)
            def f(*args):
                return param[args]
        elif d['function'] == 'constant':
            size = [len(domains[l].values()) for l in d['type']]
            weights = torch.tensor(d['weights'])
            if list(weights.size()) != size:
                raise ValueError(f'weight tensor has wrong size (expected {size}, actual {list(weights.size())}')
            def f(*args):
                return weights[args]
        else:
            raise ValueError(f'invalid factor function: {d["function"]}')
        t = tuple(g.get_node_label(l) for l in d['type'])
        g.add_terminal(EdgeLabel(name, True, t, f))

    for nt, d in j['nonterminals'].items():
        t = tuple(g.get_node_label(l) for l in d['type'])
        g.add_nonterminal(EdgeLabel(nt, False, t, None))
    g.set_start_symbol(g.get_nonterminal(j['start']))

    for r in j['rules']:
        lhs = g.get_nonterminal(r['lhs'])
        rhs = FactorGraph()
        nodes = []
        for label in r['rhs']['nodes']:
            v = Node(g.get_node_label(label))
            nodes.append(v)
            rhs.add_node(v)
        for e in r['rhs']['edges']:
            att = []
            for v in e['attachments']:
                try:
                    att.append(nodes[v])
                except IndexError:
                    raise ValueError(f'invalid attachment node {v} (out of {len(nodes)})')
            rhs.add_edge(Edge(g.get_edge_label(e['label']), att))
        ext = []
        for v in r['rhs'].get('externals', []):
            try:
                ext.append(nodes[v])
            except IndexError:
                raise ValueError(f'invalid external node {v} (out of {len(nodes)})')
        rhs.set_ext(ext)
        g.add_rule(FGGRule(lhs, rhs))
        
    return g

def fgg_to_json(g):
    """Convert an FGG to an object writable by json.dump()."""
    import domain
    import torch
    j = {}

    j['domains'] = {}
    for l in g.node_labels():
        assert l.name() == l.domain().name()
        name = l.name()
        if isinstance(l.domain(), domain.FiniteDomain):
            j['domains'][name] = {
                'class' : 'finite',
                'values' : list(l.domain().values()),
            }
        else:
            raise NotImplementedError(f'unsupported domain type {type(j.domain())}')

    j['factors'] = {}

    j['nonterminals'] = {}
    for nt in g.nonterminals():
        j['nonterminals'][nt.name()] = {
            'type': [l.name() for l in nt.type()],
        }
    j['start'] = g.start_symbol().name()
    
    j['rules'] = []
    for gr in sorted(g.all_rules(), key=lambda r: r.rule_id()):
        nodes = sorted(gr.rhs().nodes(), key=lambda v: v.node_id())
        node_nums = {v.node_id():i for (i,v) in enumerate(nodes)}
        jr = {
            'lhs': gr.lhs().name(),
            'rhs': {
                 'nodes': [v.label().name() for v in nodes],
                 'edges': [],
                'externals': [node_nums[v.node_id()] for v in gr.rhs().ext()],
            },
        }
        for e in sorted(gr.rhs().edges(), key=lambda e: e.edge_id()):
            jr['rhs']['edges'].append({
                'attachments': [node_nums[v.node_id()] for v in e.nodes()],
                'label': e.label().name(),
            })
        j['rules'].append(jr)
        
    return j

