__all__ = ['json_to_fgg', 'fgg_to_json', 'json_to_hrg', 'hrg_to_json', 'graph_to_dot', 'graph_to_tikz', 'hrg_to_tikz']

from fggs.fggs import *
from fggs import domains, factors

### JSON

def json_to_fgg(j):
    """Convert an object loaded by json.load to an FGG."""
    fgg = FGG.from_hrg(json_to_hrg(j['grammar']))

    ji = j['interpretation']
    for name, d in ji['domains'].items():
        nl = fgg.get_node_label(name)
        if d['class'] == 'finite':
            fgg.add_domain(nl, domains.FiniteDomain(d['values']))
        else:
            raise ValueError(f'invalid domain class: {d["type"]}')

    for name, d in ji['factors'].items():
        el = fgg.get_edge_label(name)
        if d['function'] == 'finite':
            weights = d['weights']
            fgg.add_factor(el, factors.FiniteFactor([fgg.domains[nl.name] for nl in el.type], weights))
        else:
            raise ValueError(f'invalid factor function: {d["function"]}')
        
    return fgg

def fgg_to_json(fgg):
    """Convert an FGG to an object writable by json.dump()."""
    jg = hrg_to_json(fgg)
    
    ji = {}

    ji['domains'] = {}
    for nl, dom in fgg.domains.items():
        if isinstance(dom, domains.FiniteDomain):
            ji['domains'][nl] = {
                'class' : 'finite',
                'values' : list(dom.values),
            }
        else:
            raise NotImplementedError(f'unsupported domain type {type(j.domain)}')

    ji['factors'] = {}
    for el, fac in fgg.factors.items():
        if isinstance(fac, factors.FiniteFactor):
            ji['factors'][el] = {
                'function': 'finite',
                'weights': fac.weights,
            }
            
    return {'grammar': jg, 'interpretation': ji}

def json_to_hrg(j):
    """Convert an object loaded by json.load to an HRG."""

    labels = {}
    for name, d in j['terminals'].items():
        t = tuple(NodeLabel(l) for l in d['type'])
        labels[name] = EdgeLabel(name, t, is_terminal=True)
    for name, d in j['nonterminals'].items():
        t = tuple(NodeLabel(l) for l in d['type'])
        labels[name] = EdgeLabel(name, t, is_nonterminal=True)

    g = HRG(labels[j['start']])
    
    for label in labels.values():
        g.add_edge_label(label)

    for r in j['rules']:
        lhs = g.get_edge_label(r['lhs'])
        rhs = Graph()
        nodes = []
        for node in r['rhs']['nodes']:
            v = Node(NodeLabel(node['label']), id=node.get('id', None))
            nodes.append(v)
            rhs.add_node(v)
        for e in r['rhs']['edges']:
            att = []
            for vi in e['attachments']:
                try:
                    att.append(nodes[vi])
                except IndexError:
                    raise ValueError(f'invalid attachment node number {vi} (out of {len(nodes)})')
            rhs.add_edge(Edge(g.get_edge_label(e['label']), att, id=e.get('id', None)))
        ext = []
        for vi in r['rhs'].get('externals', []):
            try:
                ext.append(nodes[vi])
            except IndexError:
                raise ValueError(f'invalid external node number {vi} (out of {len(nodes)})')
        rhs.ext = ext
        g.add_rule(HRGRule(lhs, rhs))
        
    return g

def hrg_to_json(g):
    """Convert an HRG to an object writable by json.dump()."""
    j = {}

    j['terminals'] = {}
    for t in g.terminals():
        j['terminals'][t.name] = {
            'type': [l.name for l in t.type],
        }
        
    j['nonterminals'] = {}
    for nt in g.nonterminals():
        j['nonterminals'][nt.name] = {
            'type': [l.name for l in nt.type],
        }
        
    j['start'] = g.start.name
    
    j['rules'] = []
    for gr in g.all_rules():
        nodes = sorted(gr.rhs.nodes(), key=lambda v: str(v.id))
        node_nums = {v:vi for vi, v in enumerate(nodes)}
        jnodes = []
        for v in nodes:
            jv = {'label': v.label.name}
            if v.persist_id:
                jv['id'] = v.id
            jnodes.append(jv)
        jr = {
            'lhs': gr.lhs.name,
            'rhs': {
                'nodes': jnodes,
                'edges': [],
                'externals': [node_nums[v] for v in gr.rhs.ext],
            },
        }
        for e in sorted(gr.rhs.edges(), key=lambda e: str(e.id)):
            je = {
                'attachments': [node_nums[v] for v in e.nodes],
                'label': e.label.name,
            }
            if e.persist_id:
                je['id'] = e.id
            jr['rhs']['edges'].append(je)
        j['rules'].append(jr)
        
    return j

def weights_to_json(weights):
    if isinstance(weights, float) or hasattr(weights, 'shape') and len(weights.shape) == 0:
        return float(weights)
    else:
        return [weights_to_json(w) for w in weights]


### GraphViz and TikZ

def _get_format(factor_formats, x, i):
    if (x is None or
        factor_formats is None or
        x.name not in factor_formats):
        return ('', str(i+1))
    fmt = factor_formats[x.name][i]
    if len(fmt) > 0 and fmt[0] in '<>^_':
        return (fmt[0], fmt[1:])
    else:
        return ('', fmt)
    
def graph_to_dot(g: Graph, factor_formats=None, lhs=None):
    """Convert a Graph to a pydot.Dot.

    factor_formats is an optional dict that provides additional
    information about factors (EdgeLabels). If f is an EdgeLabel, then
    factor_attrs[f] is a list of strs with len f.arity(). Each string
    has two optional parts:

    - '<' for an input or '>' for an output
    - a symbolic name
    """
    
    import pydot # type: ignore

    dot = pydot.Dot(graph_type='graph', rankdir='LR')
    for v in g.nodes():
        dot.add_node(pydot.Node(f'v{v.id}',
                                #label=v.label.name,
                                label='',
                                shape='circle',
                                height=0.24,
                                margin=0,
        ))
    for e in g.edges():
        if e.label.is_terminal():
            dot.add_node(pydot.Node(f'e{e.id}',
                                    label='',
                                    xlabel=e.label.name,
                                    shape='square',
                                    height=0.16,
            ))
        else:
            dot.add_node(pydot.Node(f'e{e.id}',
                                    label=e.label.name,
                                    shape='square',
                                    height=0.24,
                                    margin=0.04,
            ))
        nv = len(e.nodes)
        for i, v in enumerate(e.nodes):
            format = _get_format(factor_formats, e.label, i)
            if format[0] == '>':
                dot.add_edge(pydot.Edge(f'e{e.id}', f'v{v.id}',
                                        order=i+1,
                                        label=format[1],
                ))
            elif format[0] == '<':
                dot.add_edge(pydot.Edge(f'v{v.id}', f'e{e.id}',
                                        order=i+1,
                                        label=format[1],
                ))
            elif format[0] == '^':
                sub = pydot.Subgraph(rank='same')
                sub.add_node(pydot.Node(f'v{v.id}'))
                sub.add_node(pydot.Node(f'e{e.id}'))
                dot.add_subgraph(sub)
                dot.add_edge(pydot.Edge(f'v{v.id}', f'e{e.id}',
                                        order=i+1,
                                        label=format[1],
                ))
            elif format[0] == '_':
                sub = pydot.Subgraph(rank='same')
                sub.add_node(pydot.Node(f'v{v.id}'))
                sub.add_node(pydot.Node(f'e{e.id}'))
                dot.add_subgraph(sub)
                dot.add_edge(pydot.Edge(f'e{e.id}', f'v{v.id}',
                                        order=i+1,
                                        label=format[1],
                ))
            else:
                dot.add_edge(pydot.Edge(f'v{v.id}', f'e{e.id}',
                                        order=i+1,
                                        label=format[1],
                                        constraint=False,
                ))
    for i, v in enumerate(g.ext):
        [dv] = dot.get_node(f'v{v.id}')
        attrs = dv.get_attributes()
        attrs['ext'] = i+1
        attrs['style'] = 'filled'
        attrs['xlabel'] = _get_format(factor_formats, lhs, i)[1]
        attrs['fillcolor'] = 'black'
        
    return dot

def graph_to_tikz(g: Graph, factor_formats=None, lhs=None):
    r"""Convert a Graph to LaTeX/TikZ code.

    The resulting code makes use of several TikZ styles. Some suggested
    definitions for these styles are:

    .. code-block:: latex

        \tikzset{
          var/.style={draw,circle,fill=white,inner sep=1.5pt,minimum size=8pt},
          ext/.style={var,fill=black,text=white},
          fac/.style={draw,rectangle},
          tent/.style={font={\tiny},auto}
        }
    """
    import pydot
    
    # Convert to DOT just to get layout information
    dot = graph_to_dot(g, factor_formats, lhs)
    dot = pydot.graph_from_dot_data(dot.create_dot().decode('utf8'))[0]

    positions = {}
    def visit(d):
        for v in d.get_nodes():
            try:
                pos = v.get_attributes()['pos']
            except KeyError:
                continue
            if pos.startswith('"') and pos.endswith('"'):
                pos = pos[1:-1]
            x, y = pos.split(',', 1)
            positions[v.get_name()] = (float(x), float(y))
        for s in d.get_subgraphs():
            visit(s)
    visit(dot)

    ys = [positions[f'v{v.id}'][1] for v in g.nodes()]
    ys.extend([positions[f'e{e.id}'][1] for e in g.edges()])
    baseline = (min(ys)+max(ys))/2
    
    res = []
    res.append(rf'\begin{{tikzpicture}}[baseline={baseline}pt]')

    ext = {v.id:i for i,v in enumerate(g.ext)}
    for v in g.nodes():
        if v.id in ext:
            style = f'ext,label={_get_format(factor_formats, lhs, ext[v.id])[1]}'
        else:
            style = 'var'
        x, y = positions[f'v{v.id}']
        res.append(rf'  \node [{style}] (v{v.id}) at ({x}pt,{y}pt) {{}};')
    for e in g.edges():
        x, y = positions[f'e{e.id}']
        if e.label.is_terminal():
            res.append(rf'  \node [fac,label={{{e.label.name}}}] (e{e.id}) at ({x}pt,{y}pt) {{}};')
        else:
            res.append(rf'  \node [fac] (e{e.id}) at ({x}pt,{y}pt) {{{e.label.name}}};')
        for i, v in enumerate(e.nodes):
            label = _get_format(factor_formats, e.label, i)[1]
            res.append(rf'    \draw (e{e.id}) edge node[tent,near start] {{{label}}} (v{v.id});')
    res.append(r'\end{tikzpicture}')
    return '\n'.join(res)

def hrg_to_tikz(g, factor_formats=None):
    r"""Convert an HRG to LaTeX/TikZ code.

    The resulting code makes use of several TikZ styles. Some suggested
    definitions for these styles are:

    .. code-block:: latex

        \tikzset{
          var/.style={draw,circle,fill=white,inner sep=1.5pt,minimum size=8pt},
          ext/.style={var,fill=black,text=white},
          fac/.style={draw,rectangle},
          tent/.style={font={\tiny},auto}
        }
        """
    res = []
    res.append(r'\begin{align*}')
    for r in g.all_rules():
        # Build a little factor graph for the lhs
        lhs = Graph()
        lhs.add_edge(Edge(r.lhs(), [Node(x) for x in r.lhs.type]))
        
        res.append(graph_to_tikz(lhs, factor_formats, r.lhs) +
                   ' &\longrightarrow ' +
                   graph_to_tikz(r.rhs, factor_formats, r.lhs) + r'\\')
    res.append(r'\end{align*}')
    return '\n'.join(res)
