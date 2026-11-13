__all__ = ['json_to_fgg', 'fgg_to_json',
           'json_to_fgg_with_ring',
           'json_to_hrg', 'hrg_to_json',
           'json_to_weights',
           'tensor_to_json',
           'graph_to_dot', 'graph_to_tikz', 'hrg_to_tikz']

from itertools import repeat
from functools import reduce
from math import isfinite
from fggs.fggs import *
from fggs.indices import PhysicalAxis, productAxis, SumAxis, PatternedTensor
from fggs.semirings import Semiring, RealSemiring, ComplexSemiring
from fggs import domains, factors
import re
import torch
from torch import Tensor
from typing import cast, Tuple
from enum import Enum

class WeightType(Enum):
    REAL = 1
    COMPLEX = 2

### JSON

def weight_type_fold(w1, w2):
    if w1 == WeightType.REAL:
        return w2
    else:
        return w1

def json_weights_type(j):
    weight_type = WeightType.REAL

    ji = j['interpretation']
    for name, d in ji['factors'].items():
        if d['function'] == 'constant':
            weight_type = weight_type_fold(weight_type, json_scalar_type(d['weight']))
        elif d['function'] == 'finite':
            weight_type = weight_type_fold(weight_type, json_scalar_type(d['weights']))
        else:
            raise ValueError(f'invalid factor function: {d["function"]}')

    dtype = torch.get_default_dtype()
    if weight_type == WeightType.REAL:
        if not (dtype == torch.float32 or dtype == torch.float64):
            raise ValueError(f'The given default datatype of torch is not supported: {dtype}')
    else:
        if torch.get_default_dtype() == torch.float32:
            dtype = torch.cfloat
        elif torch.get_default_dtype() == torch.float64:
            dtype = torch.cdouble
        else:
            raise ValueError(f'The given default datatype of torch is not supported: {dtype}')

    return dtype


def json_semiring(ring_type: torch.dtype) -> Semiring:
    if ring_type == torch.float32 or ring_type == torch.float64:
        return RealSemiring(dtype=ring_type)
    elif ring_type == torch.cfloat or ring_type == torch.cdouble:
        return ComplexSemiring(dtype=ring_type)
    else:
        raise ValueError(f'Unsupported semiring type: {ring_type}')


def json_is_number(j):
    return (isinstance(j, float) or isinstance(j, int))


def json_is_complex(j):
    return (isinstance(j, dict) and
            sorted(j.keys()) == ['im', 're'] and
            json_is_number(j['im']) and
            json_is_number(j['re']))


def json_scalar_type(j):
    if isinstance(j, dict) and "physical" in j:
        physical = j["physical"]
    else:
        physical = j
    if json_is_number(physical):
        return WeightType.REAL
    elif json_is_complex(physical):
        return WeightType.COMPLEX
    elif isinstance(physical, list):
        return reduce(weight_type_fold, (json_scalar_type(i) for i in physical), WeightType.REAL)
    else:
        raise ValueError(f'not a valid scalar value type: {j}')

def json_to_fgg(j):
    """Convert an object loaded by json.load to an FGG."""
    dtype = json_weights_type(j)
    return json_to_fgg_with_dtype(j, dtype)


def json_to_fgg_with_dtype(j, dtype):
    """Convert an object loaded by json.load to an FGG with the given dtype."""

    fgg = FGG.from_hrg(json_to_hrg(j['grammar']))

    ji = j['interpretation']
    for name, d in ji['domains'].items():
        if not fgg.has_node_label_name(name):
            fgg.add_node_label(NodeLabel(name))
        nl = fgg.get_node_label(name)
        if d['class'] == 'finite':
            fgg.add_domain(nl, domains.FiniteDomain(d['values']))
        elif d['class'] == 'range':
            fgg.add_domain(nl, domains.RangeDomain(d['size']))
        else:
            raise ValueError(f'invalid domain class: {d["type"]}')

    for name, d in ji['factors'].items():
        el = fgg.get_edge_label(name)
        doms = [fgg.domains[nl.name] for nl in el.type]
        if d['function'] == 'constant':
            fgg.add_factor(el, factors.ConstantFactor(doms, json_to_scalar(d['weight'])))
        elif d['function'] == 'finite':
            weights = json_to_weights(d['weights'], dtype)
            fgg.add_factor(el, factors.FiniteFactor(doms, weights))
        else:
            raise ValueError(f'invalid factor function: {d["function"]}')
        
    return fgg


def json_to_fgg_with_ring(j) -> Tuple[FGG, Semiring]:
    ring_type = json_weights_type(j)
    ring = json_semiring(ring_type)
    fgg = json_to_fgg_with_dtype(j, ring_type)
    return fgg, ring


def fgg_to_json(fgg):
    """Convert an FGG to an object writable by json.dump()."""
    jg = hrg_to_json(fgg)
    
    ji = {}

    ji['domains'] = {nl: dom.to_json() for nl, dom in fgg.domains.items()}
    ji['factors'] = {el: fac.to_json() for el, fac in fgg.factors.items()}
            
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
        nodes = gr.rhs.nodes()
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
        for e in gr.rhs.edges():
            je = {
                'attachments': [node_nums[v] for v in e.nodes],
                'label': e.label.name,
            }
            if e.persist_id:
                je['id'] = e.id
            jr['rhs']['edges'].append(je)
        j['rules'].append(jr)
        
    return j

def json_to_scalar(j):
    if json_is_complex(j):
        return complex(j['re'], j['im'])
    else:
        return j

def json_to_array(j):
    if isinstance(j, list):
        return [json_to_array(s) for s in j]
    else:
        return json_to_scalar(j)

def json_to_tensor(j, dtype):
    return torch.tensor(json_to_array(j), dtype=dtype)

def json_to_weights(j, dtype):
    """Convert an object loaded by json.load to an PatternedTensor."""
    if isinstance(j, dict) and "physical" in j:
        physical = json_to_tensor(j["physical"], dtype)
        expand = j.get("expand")
        if expand:
            physical = physical.expand([*expand, *repeat(-1, physical.ndim)])
        paxes = tuple(PhysicalAxis(n) for n in physical.size())
        vaxes = j.get("vaxes")
        if vaxes is not None:
            vaxes = tuple(json_to_axis(r, paxes) for r in vaxes)
        default = j.get("default", 0.)
        return PatternedTensor(physical, paxes, vaxes, default)
    else:
        physical = json_to_tensor(j, dtype)
        return PatternedTensor(physical)

def scalar_to_json(s : Tensor):
    if s.is_complex():
        {'re': s.item().real, 'im': s.item().imag}
    else:
        return s.item()

def tensor_to_json(s : Tensor):
    if s.ndim == 0:
        return scalar_to_json(s)
    else:
        return [tensor_to_json(e) for e in s]

def weights_to_dict_json(fgg : FGG, edge_label : EdgeLabel, weights : Tensor):
    def to_dict(edge_type, weights):
        if len(edge_type) == 0:
            return scalar_to_json(weights)
        else:
            d = fgg.domains[edge_type[0].name]
            if isinstance(d, domains.FiniteDomain) and d.size() == len(weights):
                return {d.denumberize(i):
                        to_dict(edge_type[1:], w)
                        for i, w in enumerate(weights)}
            else:
                return [to_dict(edge_type[1:], w)
                        for w in weights]
    return to_dict(edge_label.type, weights)

def json_to_axis(r, env):
    if isinstance(r, list):
        return productAxis(json_to_axis(r1, env) for r1 in r)
    elif isinstance(r, dict):
        return SumAxis(r["before"],
                       json_to_axis(r["term"], env),
                       r["after"])
    else:
        return env[r]



### GraphViz and TikZ
def escape_to_html(in_str):
    table = {"&": "&amp;",
             "<": "&lt;",
             ">": "&gt;",
             "\"": "&quot;",
             "'": "&#39;"}
    result = ""
    for s in in_str:
        if s in table:
            result += table[s]
        else:
            result += s
    return result


def escape_to_latex(in_str):
    table = {"\\": "\\textbackslash{}",
             "_": "\\_",
             "^": "\\textasciicircum{}",
             "<": "\\textless{}",
             ">": "\\textgreater{}",
             "{": "\\{",
             "}": "\\}"}
    result = ""
    for s in in_str:
        if s in table:
            result += table[s]
        else:
            result += s
    return result


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
        node = pydot.Node(f'v{v.id}',
                          label='',
                          shape='circle',
                          height=0.24,
                          margin=0,
                          )
        node.set_name(f'v{v.id}')
        dot.add_node(node)
    for e in g.edges():
        if e.label.is_terminal:
            node = pydot.Node(f'e{e.id}',
                              label='',
                              # xlabel=escape_to_html(e.label.name),
                              shape='square',
                              height=0.16,
                              )
            node.set_name(f'e{e.id}')
            dot.add_node(node)
        else:
            node = pydot.Node(f'e{e.id}',
                              # label=escape_to_html(e.label.name),
                              shape='square',
                              height=0.24,
                              margin=0.04,
                              )
            node.set_name(f'e{e.id}')
            dot.add_node(node)
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
    # print(dot.to_string())
    graph = pydot.graph_from_dot_data(dot.create_dot().decode('utf8'))
    if graph:
        dot = graph[0]
    else:
        raise ValueError(f'Failed to generate a tikz from a graph')

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
            name = re.sub(r'"(.*)"', r'\1', v.get_name())
            positions[name] = (float(x), float(y))
        for s in d.get_subgraphs():
            visit(s)
    visit(dot)

    ys = [positions[f'v{v.id}'][1] for v in g.nodes()]
    ys.extend([positions[f'e{e.id}'][1] for e in g.edges()])
    baseline = (min(ys)+max(ys))/2
    
    res = []
    res.append(rf'\begin{{tikzpicture}}[baseline={baseline}pt]')

    ext = {v.id:i for i,v in enumerate(g.ext)}
    for i,v in enumerate(g.nodes()):
        if v.id in ext:
            style = f'ext,label={escape_to_latex(_get_format(factor_formats, lhs, ext[v.id])[1])}'
        else:
            style = 'var'
        x, y = positions[f'v{v.id}']
        res.append(rf'  \node [{style},label={{[right]{escape_to_latex(v.label.name)}}}] (v{v.id}) at ({x+20}pt,{y}pt) {{}};')
    for e in g.edges():
        x, y = positions[f'e{e.id}']
        if e.label.is_terminal:
            res.append(rf'  \node [fac,label={{[right]{{T,{escape_to_latex(e.label.name)}}}}}] (e{e.id}) at ({x}pt,{y}pt) {{}};')
        else:
            res.append(rf'  \node [fac,label={{[right]{{N,{escape_to_latex(e.label.name)}}}}}] (e{e.id}) at ({x}pt,{y}pt) {{}};')
        for i, v in enumerate(e.nodes):
            label = _get_format(factor_formats, e.label, i)[1]
            res.append(rf'  \draw (e{e.id}) edge node[tent,very near start] {{{escape_to_latex(label)}}} (v{v.id});')
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
    res.append(r"""\documentclass[multi=page,preview]{standalone}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{tikz}
\usetikzlibrary{calc}

\tikzset{
  var/.style={draw,circle,fill=white,inner sep=1.5pt,minimum size=8pt},
  ext/.style={var,fill=black,text=white},
  fac/.style={draw,rectangle},
  tent/.style={font={\tiny},auto}
}

\begin{document}""")
    for r in g.all_rules():
        # Build a little factor graph for the lhs
        lhs = Graph()
        ext_nodes = [Node(x) for x in r.lhs.type]
        lhs.add_edge(Edge(r.lhs, ext_nodes))
        lhs.ext = ext_nodes
        
        res.append(r'\begin{page}')
        res.append(r'\begin{align*}')
        res.append(graph_to_tikz(lhs, factor_formats, r.lhs) +
                   ' &\\longrightarrow ' +
                   graph_to_tikz(r.rhs, factor_formats, r.lhs) + r'\\')
        res.append(r'\end{align*}')
        res.append(r'\end{page}')
    res.append(r'\end{document}')
    print(f'{len(g.all_rules())} processed.')
    return '\n'.join(res)
