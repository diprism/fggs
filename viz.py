import fgg_representation as fggs
import pydot

def factorgraph_to_dot(g: fggs.FactorGraph):
    """Convert a FactorGraph to a pydot.Dot."""
    
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
        for i, v in enumerate(e.nodes(), start=1):
            # Although the tentacles are not directed, dot's layout algorithm
            # does use their direction. Reverse the direction of the last-numbered
            # tentacle to make it appear to the right of the other tentacles.
            if i == nv:
                dot.add_edge(pydot.Edge(f'e{e.edge_id()}', f'v{v.node_id()}',
                                        order=i,
                                        label=i,
                ))
            else:
                dot.add_edge(pydot.Edge(f'v{v.node_id()}', f'e{e.edge_id()}',
                                        order=i,
                                        label=i,
                ))
    for i, v in enumerate(g.ext(), start=1):
        [dv] = dot.get_node(f'v{v.node_id()}')
        attrs = dv.get_attributes()
        attrs['ext'] = i
        attrs['style'] = 'filled'
        attrs['label'] = i
        attrs['fontcolor'] = 'white'
        attrs['fillcolor'] = 'black'
        
    return dot

def factorgraph_to_tikz(g: fggs.FactorGraph):
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
    
    dot = factorgraph_to_dot(g)
    # Add layout information
    dot = pydot.graph_from_dot_data(dot.create_dot().decode('utf8'))[0]

    def get_pos(name):
        [dv] = dot.get_node(name)
        pos = dv.get_attributes()['pos']
        if pos.startswith('"') and pos.endswith('"'):
            pos = pos[1:-1]
        x, y = pos.split(',', 1)
        return float(x), float(y)

    ys = [get_pos(f'v{v.node_id()}')[1] for v in g.nodes()]
    ys.extend([get_pos(f'e{e.edge_id()}')[1] for e in g.edges()])
    baseline = (min(ys)+max(ys))/2
    
    res = []
    res.append(rf'\begin{{tikzpicture}}[baseline={baseline}pt]')

    ext = {v.node_id():i for i,v in enumerate(g.ext(), start=1)}
    for v in g.nodes():
        vid = v.node_id()
        if vid in ext:
            style = 'ext'
            label = ext[vid]
        else:
            style = 'var'
            label = ''
        x, y = get_pos(f'v{vid}')
        res.append(rf'  \node [{style}] (v{vid}) at ({x}pt,{y}pt) {{{label}}};')
    for e in g.edges():
        x, y = get_pos(f'e{e.edge_id()}')
        if e.label().is_terminal():
            res.append(rf'  \node [fac,label={{{e.label().name()}}}] (e{e.edge_id()}) at ({x}pt,{y}pt) {{}};')
        else:
            res.append(rf'  \node [fac] (e{e.edge_id()}) at ({x}pt,{y}pt) {{{e.label().name()}}};')
        for i, v in enumerate(e.nodes(), start=1):
            res.append(rf'    \draw (e{e.edge_id()}) edge node[tent,near start] {{{i}}} (v{v.node_id()});')
    res.append(r'\end{tikzpicture}')
    return '\n'.join(res)

def fgg_to_tikz(g):
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
        res.append(f'\mbox{{{r.lhs().name()}}} &\longrightarrow ' + factorgraph_to_tikz(r.rhs()) + r'\\')
    res.append(r'\end{align*}')
    return '\n'.join(res)

if __name__ == "__main__":
    import hmm
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
              fgg_to_tikz(hmm.hmm) + '\n' +
              r'\end{document}', file=outfile)
