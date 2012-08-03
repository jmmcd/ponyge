#!/usr/bin/env python

import random
import copy

# Derivation trees for PonyGE. The original PonyGE implemented GE
# proper, that is integer-array genomes for mutation and crossover,
# and derivation from the genomes. The aim here is to do away with the
# integer genome and just use derivation trees as genomes. We can
# randomly-generate DTs, and cross them over and mutate them.

# We use two main ideas here. First, a tree is represented as a list
# whose first element is a symbol and whose other elements are other
# trees. A class representing a tree or a treenode is not needed
# here. Trees already know how to print themselves nicely.

# Second, every node in a tree has a "path", that is a tuple of
# integers. The tree itself is accessed with the empty tuple. The root
# node of a tree is accessed as (0,). The first subtree of the root is
# (1,), and (2,) is the second subtree of the root. Getting (2, 0)
# gets the root of that subtree. Given a node's path we get its
# *parent's subtree's path* by lopping off the last *two* elements.

# TODO avoid crossing-over identical subtrees
# TODO avoid mutating-in identical subtree
# TODO implement a max-depth (we have the depth value of every node)
# TODO should crossover/mutation work on terminals?
# TODO set the number of nodes in the tree as the number of codons for ponyge
# TODO how should the experiments go -- vanilla ponyge v derivation tree, or should they start from the same place (ie use same initialisation)
# TODO tests

def path(item):
    return item[2]

def get_node(t, path):
    """Given a tree and a path, return the node at that path."""
    s = get_subtree(t, path)
    if isinstance(s, str):
        return s
    else:
        return s[0]

def get_subtree(t, path):
    """Given a tree and a path, return the subtree at that path."""
    for item in path:
        t = t[item]
    return t
    
def depth(path):
    """The depth of any node is the number on nonzero elements in its
    path."""
    return len([el for el in path if el != 0])

def derived_str(dt, grammar):
    """Get the derived string."""
    return "".join([s[0] for s in traverse(dt) if s[0] in grammar.terminals])

def traverse(t, path=None):
    """Depth-first traversal of the tree t, yielding at each step the
    node, the subtree rooted at that node, and the path. The path
    passed-in is the "path so far"."""
    if path is None: path = tuple()
    yield t[0], t, path + (0,)
    for i, item in enumerate(t[1:], start=1):
        if isinstance(item, str):
            yield item, item, path + (i,)
        else:
            for s in traverse(item, path + (i,)):
                yield s

def random_dt(grammar, s=None):
    """Recursively create a random derivation tree given a start
    symbol and a grammar. Please be amazed at how easy it is to do a
    derivation when we do away with the integer genome."""
    if s in grammar.terminals: return s
    if s is None: s=grammar.start_rule[0]
    prod = random.choice(grammar.rules[s])
    return [s] + [random_dt(grammar, s[0]) for s in prod]

def dt_mutation(t, grammar):
    """Given a derivation tree, mutate it by choosing a non-terminal
    and growing from there according to the grammar."""
    
    # Get all the items in t, excluding the root and terminals.
    m_pts = [p for p in list(traverse(t))[1:]
             if p[0] not in grammar.terminals]

    # Choose a point *uniformly among the points*
    m_pt = random.choice(m_pts)

    # Get the *subtree of the parent* (ie traverse up to path element
    # -2) of the mutation point, to allow insertion. See comment in
    # dt_crossover().
    m_pt_pth = path(m_pt)
    assert len(m_pt_pth) >= 2
    m_ins_pt = get_subtree(t, m_pt_pth[:-2])

    # Get the index in the parent list at which to paste
    m_idx = m_pt_pth[-2]

    # Get the new random subtree to be pasted
    m_subtree = random_dt(grammar, m_pt[0])
    
    # Perform the mutation
    m_ins_pt[m_idx] = m_subtree

def dt_crossover(t, s, grammar):
    """Cross two trees over, but only on matching non-terminals."""
    
    # Get all the items in t and in s, excluding the root and
    # terminals.
    t_pts = [p for p in list(traverse(t))[1:]
             if p[0] not in grammar.terminals]
    s_pts = [p for p in list(traverse(s))[1:]
             if p[0] not in grammar.terminals]

    # Get their labels
    t_lbls = set([p[0] for p in t_pts])
    s_lbls = set([p[0] for p in s_pts])

    # Get the labels which are in both parents
    x_lbls = (t_lbls & s_lbls)

    # Choose a label *uniformly among the labels*
    x_lbl = random.choice(list(x_lbls))

    # Find the points in t and in s which have that label
    tx_pts = [p for p in t_pts if p[0] == x_lbl]
    sx_pts = [p for p in s_pts if p[0] == x_lbl]

    # Choose the crossover points
    tx_pt = random.choice(tx_pts)
    sx_pt = random.choice(sx_pts)

    # Get the *subtree of the parent* (ie traverse up to path element
    # -2) of the crossover points, to allow insertion. We are safe in
    # doing this because there are only two types of *nodes* which
    # have less than 2 elements in their path: the root, and any
    # terminals which are children of the root. We won't ever be
    # crossing-over at those points.
    tx_pt_pth = path(tx_pt)
    sx_pt_pth = path(sx_pt)
    assert len(tx_pt_pth) >= 2
    assert len(sx_pt_pth) >= 2
    t_ins_pt = get_subtree(t, tx_pt_pth[:-2])
    s_ins_pt = get_subtree(s, sx_pt_pth[:-2])

    # Get the index in the parent list at which to paste
    t_idx = tx_pt_pth[-2]
    s_idx = sx_pt_pth[-2]
    
    # Get the subtrees to be pasted
    tx_subtree = tx_pt[1]
    sx_subtree = sx_pt[1]

    # Perform the crossover
    t_ins_pt[t_idx], s_ins_pt[s_idx] = sx_subtree, tx_subtree

def main():
    from ponyge import Grammar
    grammar = Grammar("grammars/letter.bnf")
    
    dt = random_dt(grammar)
    print "dt", dt
    print "dt derivation", derived_str(dt, grammar)

    print "subtree (1,)", get_subtree(dt, (1,))
    print "node (1,)", get_node(dt, (1,))
    print "subtree (1, 1)", get_subtree(dt, (1, 1))
    print "node (1, 1)", get_node(dt, (1, 1))
    print "depth of node (1, 1)", depth((1, 1))
    
    ds = random_dt(grammar)
    print "ds", ds
    print "ds derivation", derived_str(ds, grammar)
    
    dt_crossover(dt, ds, grammar)
    print derived_str(dt, grammar)

    dt_mutation(dt, grammar)
    print derived_str(dt, grammar)

    print "++++++++++++++++++"
    for item in traverse(dt):
        print item

if __name__ == "__main__":
    main()
