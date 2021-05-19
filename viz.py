import fgg_representation as fggs
import pydot

def _get_format(factor_formats, x, i):
    if (x is None or
        factor_formats is None or
        x.name() not in factor_formats):
        return ('', str(i+1))
    fmt = factor_formats[x.name()][i]
    if len(fmt) > 0 and fmt[0] in '<>^_':
        return (fmt[0], fmt[1:])
    else:
        return ('', fmt)
    
def factorgraph_to_dot(g: fggs.FactorGraph, factor_formats=None, lhs=None):
    """Convert a FactorGraph to a pydot.Dot.

    factor_formats is an optional dict that provides additional
    information about factors (EdgeLabels). If f is an EdgeLabel, then
    factor_attrs[f] is a list of strs with len f.arity(). Each string
    has two optional parts:

    - '<' for an input or '>' for an output
    - a symbolic name
    """

    dot = pydot.Dot(graph_type='graph', rankdir='LR')
    for v in g.nodes():
        dot.add_node(pydot.Node(f'v{v.node_id()}',
                                #label=v.label().name(),
                                label='',
                                shape='circle',
                                height=0.24,
                                margin=0,
        ))
    for e in g.edges():
        if e.label().is_terminal():
            dot.add_node(pydot.Node(f'e{e.edge_id()}',
                                    label='',
                                    xlabel=e.label().name(),
                                    shape='square',
                                    height=0.16,
            ))
        else:
            dot.add_node(pydot.Node(f'e{e.edge_id()}',
                                    label=e.label().name(),
                                    shape='square',
                                    height=0.24,
                                    margin=0.04,
            ))
        nv = len(e.nodes())
        for i, v in enumerate(e.nodes()):
            format = _get_format(factor_formats, e.label(), i)
            if format[0] == '>':
                dot.add_edge(pydot.Edge(f'e{e.edge_id()}', f'v{v.node_id()}',
                                        order=i+1,
                                        label=format[1],
                ))
            elif format[0] == '<':
                dot.add_edge(pydot.Edge(f'v{v.node_id()}', f'e{e.edge_id()}',
                                        order=i+1,
                                        label=format[1],
                ))
            elif format[0] == '^':
                sub = pydot.Subgraph(rank='same')
                sub.add_node(pydot.Node(f'v{v.node_id()}'))
                sub.add_node(pydot.Node(f'e{e.edge_id()}'))
                dot.add_subgraph(sub)
                dot.add_edge(pydot.Edge(f'v{v.node_id()}', f'e{e.edge_id()}',
                                        order=i+1,
                                        label=format[1],
                ))
            elif format[0] == '_':
                sub = pydot.Subgraph(rank='same')
                sub.add_node(pydot.Node(f'v{v.node_id()}'))
                sub.add_node(pydot.Node(f'e{e.edge_id()}'))
                dot.add_subgraph(sub)
                dot.add_edge(pydot.Edge(f'e{e.edge_id()}', f'v{v.node_id()}',
                                        order=i+1,
                                        label=format[1],
                ))
            else:
                dot.add_edge(pydot.Edge(f'v{v.node_id()}', f'e{e.edge_id()}',
                                        order=i+1,
                                        label=format[1],
                                        constraint=False,
                ))
    for i, v in enumerate(g.ext()):
        [dv] = dot.get_node(f'v{v.node_id()}')
        attrs = dv.get_attributes()
        attrs['ext'] = i+1
        attrs['style'] = 'filled'
        attrs['xlabel'] = _get_format(factor_formats, lhs, i)[1]
        attrs['fillcolor'] = 'black'
        
    return dot

def factorgraph_to_tikz(g: fggs.FactorGraph, factor_formats=None, lhs=None):
    """Convert a FactorGraph to LaTeX/TikZ code.

    The resulting code makes use of several TikZ styles. Some suggested
    definitions for these styles are:

    \tikzset{
      var/.style={draw,circle,fill=white,inner sep=1.5pt,minimum size=8pt},
      ext/.style={var,fill=black,text=white},
      fac/.style={draw,rectangle},
      tent/.style={font={\tiny},auto}
    }
    """
    
    # Convert to DOT just to get layout information
    dot = factorgraph_to_dot(g, factor_formats, lhs)
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

    ys = [positions[f'v{v.node_id()}'][1] for v in g.nodes()]
    ys.extend([positions[f'e{e.edge_id()}'][1] for e in g.edges()])
    baseline = (min(ys)+max(ys))/2
    
    res = []
    res.append(rf'\begin{{tikzpicture}}[baseline={baseline}pt]')

    ext = {v.node_id():i for i,v in enumerate(g.ext())}
    for v in g.nodes():
        vid = v.node_id()
        if vid in ext:
            style = f'ext,label={_get_format(factor_formats, lhs, ext[vid])[1]}'
        else:
            style = 'var'
        x, y = positions[f'v{vid}']
        res.append(rf'  \node [{style}] (v{vid}) at ({x}pt,{y}pt) {{}};')
    for e in g.edges():
        x, y = positions[f'e{e.edge_id()}']
        if e.label().is_terminal():
            res.append(rf'  \node [fac,label={{{e.label().name()}}}] (e{e.edge_id()}) at ({x}pt,{y}pt) {{}};')
        else:
            res.append(rf'  \node [fac] (e{e.edge_id()}) at ({x}pt,{y}pt) {{{e.label().name()}}};')
        for i, v in enumerate(e.nodes()):
            label = _get_format(factor_formats, e.label(), i)[1]
            res.append(rf'    \draw (e{e.edge_id()}) edge node[tent,near start] {{{label}}} (v{v.node_id()});')
    res.append(r'\end{tikzpicture}')
    return '\n'.join(res)

def fgg_to_tikz(g, factor_formats=None):
    """Convert an FGG to LaTeX/TikZ code.

    The resulting code makes use of several TikZ styles. Some suggested
    definitions for these styles are:

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
        lhs = fggs.FactorGraph()
        lhs.add_edge(fggs.Edge(r.lhs(), [fggs.Node(x) for x in r.lhs().type()]))
        
        res.append(factorgraph_to_tikz(lhs, factor_formats, r.lhs()) +
                   ' &\longrightarrow ' +
                   factorgraph_to_tikz(r.rhs(), factor_formats, r.lhs()) + r'\\')
    res.append(r'\end{align*}')
    return '\n'.join(res)

if __name__ == "__main__":
    import hmm

    factor_formats = {
        'S': [],
        'X': ['<prev'],
        'BOS': ['>'],
        'EOS': ['<'],
        'Ttable': ['<prev', '>cur'],
        'Etable': ['^tag', '_word'],
    }

    with open('viz.tex', 'w') as outfile:
        print(r'''
\documentclass{article}
\usepackage{tikz}
\tikzset{
  var/.style={draw,circle,fill=white,inner sep=1.5pt,minimum size=8pt},
  ext/.style={var,fill=black,text=white},
  fac/.style={draw,rectangle},
  tent/.style={font={\tiny},auto}
}

\usepackage{amsmath}
\begin{document}''' +
              fgg_to_tikz(hmm.hmm, factor_formats) + '\n' +
              r'\end{document}', file=outfile)
