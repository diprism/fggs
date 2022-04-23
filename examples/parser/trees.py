import re
import itertools

class RootDeletedException(Exception):
    pass

class Node:
    def __init__(self, label, children=None):
        self.label = label
        self.children = children or []
        for (i,child) in enumerate(self.children):
            if child.parent is not None:
                child.detach()
            child.parent = self
            child.order = i
        self.parent = None
        self.order = 0

    def __str__(self):
        return self.label

    def _pretty_print(self):
        p = PrettyPrinter()
        for child in self.children:
            p.append(child._pretty_print())
        p.root(self.label)
        return p

    def _subtree_str(self):
        if len(self.children) != 0:
            return "(%s %s)" % (self.label, " ".join(child._subtree_str() for child in self.children))
        else:
            s = '%s' % self.label
            #s = s.replace("(", "-LRB-")
            #s = s.replace(")", "-RRB-")
            return s

    def insert_child(self, i, child):
        if child.parent is not None:
            child.detach()
        child.parent = self
        self.children[i:i] = [child]
        for j in range(i,len(self.children)):
            self.children[j].order = j

    def append_child(self, child):
        if child.parent is not None:
            child.detach()
        child.parent = self
        self.children.append(child)
        child.order = len(self.children)-1

    def delete_child(self, i):
        self.children[i].parent = None
        self.children[i].order = 0
        self.children[i:i+1] = []
        for j in range(i,len(self.children)):
            self.children[j].order = j

    def detach(self):
        if self.parent is None:
            raise RootDeleteException
        self.parent.delete_child(self.order)

    def delete_clean(self):
        "Cleans up childless ancestors"
        parent = self.parent
        self.detach()
        if len(parent.children) == 0:
            parent.delete_clean()

    def bottomup(self):
        for child in self.children:
            for node in child.bottomup():
                yield node
        yield self

    def leaves(self):
        if len(self.children) == 0:
            yield self
        else:
            for child in self.children:
                for leaf in child.leaves():
                    yield leaf

class Tree:
    def __init__(self, root):
        self.root = root

    def __str__(self):
        return self.root._subtree_str()

    def pretty_print(self):
        p = self.root._pretty_print()
        return str(p)

    interior_node = re.compile(r"\s*\(([^\s)]*)")
    close_brace = re.compile(r"\s*\)")
    leaf_node = re.compile(r'\s*([^\s)]+)')

    @staticmethod
    def _scan_tree(s):
        result = Tree.interior_node.match(s)
        if result != None:
            label = result.group(1)
            pos = result.end()
            children = []
            (child, length) = Tree._scan_tree(s[pos:])
            while child != None:
                children.append(child)
                pos += length
                (child, length) = Tree._scan_tree(s[pos:])
            result = Tree.close_brace.match(s[pos:])
            if result != None:
                pos += result.end()
                return Node(label, children), pos
            else:
                return (None, 0)
        else:
            result = Tree.leaf_node.match(s)
            if result != None:
                pos = result.end()
                label = result.group(1)
                #label = label.replace("-LRB-", "(")
                #label = label.replace("-RRB-", ")")
                return (Node(label,[]), pos)
            else:
                return (None, 0)

    @staticmethod
    def from_str(s):
        s = s.strip()
        (tree, n) = Tree._scan_tree(s)
        if tree is None:
            return None
        return Tree(tree)

    def bottomup(self):
        """ Traverse the nodes of the tree bottom-up. """
        return self.root.bottomup()

    def leaves(self):
        """ Traverse the leaf nodes of the tree. """
        return self.root.leaves()

    def remove_empty(self):
        """ Remove empty nodes. """
        nodes = list(self.bottomup())
        for node in nodes:
            if node.label == '-NONE-':
                try:
                    node.delete_clean()
                except RootDeletedException:
                    self.root = None

class PrettyPrinter:
    def __init__(self):
        self.roots = []
        self.lines = [] # (left margin, string)

    def width(self):
        if len(self.lines) == 0:
            return 0
        else:
            return max(left+len(string) for (left, string) in self.lines)

    def root(self, string):
        if len(self.roots) == 0:
            self.roots = [len(string)//2]
            self.lines[0:0] = [(0, string)]
            return
            
        w = self.width()

        newroot = (self.roots[0] + self.roots[-1])//2
        nodeline = (newroot-(len(string)-1)//2, string)

        if nodeline[0] < 0:
            shift = -nodeline[0]
            nodeline = (0, string)
            self.lines = [(left+shift, s) for (left, s) in self.lines]
            self.roots = [r+shift for r in self.roots]

        if len(self.roots) == 1:
            edgeline = (self.roots[0], '│')
            self.lines[0:0] = [nodeline, edgeline]
        elif len(self.roots) > 1:
            top = ['─'] * (self.roots[-1]-self.roots[0]+1)
            top[0] = '┌'
            for i in range(1, len(self.roots)-1):
                top[self.roots[i]-self.roots[0]] = '┬'
            top[self.roots[-1]-self.roots[0]] = '┐'
            i = newroot-self.roots[0]
            top[i] = {'─':'┴', '┬':'┼', '┌':'├', '┐':'┤'}[top[i]]
            edgeline = (self.roots[0], ''.join(top))
            self.roots = [newroot]
            self.lines[0:0] = [nodeline, edgeline]
        
    def append(self, other):
        w = self.width()

        # How close can we put them without a collision?
        minspace = None
        for (sline, oline) in zip(self.lines, other.lines):
            sleft, sstring = sline
            oleft, ostring = oline
            space = w-sleft-len(sstring) + oleft
            if minspace is None or space < minspace:
                minspace = space
        if minspace is None: minspace = 0
        
        if w > 0:
            offset = -minspace+1
        else:
            offset = 0
        offset = max(-1, offset)
        
        new = []
        for (sline, oline) in itertools.zip_longest(self.lines, other.lines, fillvalue=None):
            if sline is None:
                oleft, ostring = oline
                new.append((w+offset+oleft, ostring))
            elif oline is None:
                new.append(sline)
            else:
                sleft, sstring = sline
                sright = w - sleft-len(sstring)
                oleft, ostring = oline
                new.append((sleft, sstring + ' '*(sright + offset + oleft) + ostring))
        self.lines = new
        self.roots.extend([w+offset+r for r in other.roots])

    def __str__(self):
        ret = []
        for (left, string) in self.lines:
            ret.append(' '*left + string)
        return '\n'.join(ret)

if __name__ == "__main__":
    import sys
    import re
    for line in sys.stdin:
        t = Tree.from_str(line)
        if t.root is not None:
            t.root.label = 'TOP'
            t.remove_empty()
            for n in t.bottomup():
                if len(n.children) == 0: continue
                n.label = re.sub(r'^([^ -]+)-.*', r'\1', n.label)
            print(t)
