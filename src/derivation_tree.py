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
# gets the root of that subtree.

# TODO avoid crossing-over identical subtrees
# TODO avoid mutating-in identical subtree
# TODO implement a max-depth (we have the depth value of every node)
# TODO allow crossover at start symbol, but not if both xover pts are root
# TODO should crossover/mutation work on terminals?
# TODO set the number of nodes in the tree as the number of codons for ponyge
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
    
def is_terminal(s):
    """Is a string a terminal?"""
    return not (s.startswith("<") and s.endswith(">"))

def depth(item):
    """The depth of any node is the length of its path, minus 1."""
    return len(path(item)) - 1

def derived_str(dt):
    """Get the derived string."""
    return "".join([s[0] for s in traverse(dt) if is_terminal(s[0])])

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
    if s is None: s=grammar.start_rule[0]
    if is_terminal(s): return s
    prod = random.choice(grammar.rules[s])
    return [s] + [random_dt(grammar, s[0]) for s in prod]

def dt_mutation(t, grammar):
    """Given a derivation tree, mutate it by choosing a non-terminal
    and growing from there according to the grammar."""
    
    # Get all the items in t, excluding the start symbol and terminals
    t_pts = [p for p in traverse(t)
             if p[0] not in grammar.terminals
             and p[0] != grammar.start_rule[0]]

    # Choose a point uniformly among the points.
    m_pt = random.choice(t_pts)    

    # Get the *grandparents* (ie traverse up to path element -2) of
    # the mutation point, to allow insertion. See comment in dt_crossover().
    t_ins_pt = get_subtree(t, path(m_pt)[:-2])
    
    # Perform the mutation: it's just a new random_dt()
    t_ins_pt[path(m_pt)[-2]] = random_dt(grammar, m_pt[0])

def dt_crossover(t, s, grammar):
    """Cross two trees over, but only on matching non-terminals."""
    
    # Get all the items in t and in s
    t_pts = list(traverse(t))
    s_pts = list(traverse(s))

    # Get their labels
    t_lbls = [p[0] for p in t_pts]
    s_lbls = [p[0] for p in s_pts]

    # Get the labels which are in both parents, excluding the start
    # symbol and terminals
    x_lbls = [lbl for lbl in t_lbls if lbl in s_lbls
              and lbl not in grammar.terminals
              and lbl != grammar.start_rule[0]]

    # Choose a label uniformly among the labels.
    x_lbl = random.choice(x_lbls)

    # Find the points in t and in s which have that label
    tx_pts = [p for p in t_pts if p[0] == x_lbl]
    sx_pts = [p for p in s_pts if p[0] == x_lbl]

    # Choose the crossover points
    tx_pt = random.choice([p for p in tx_pts if p[0] == x_lbl])
    sx_pt = random.choice([p for p in sx_pts if p[0] == x_lbl])

    # Get the *grandparents* (ie traverse up to path element -2) of
    # the crossover points, to allow insertion. This is
    # counter-intuitive: you would expect to get the *parent*
    # here. Note that a node's parent is the first element of the same
    # sublist. The grandparent is the containing list, and that's what
    # we need to insert into.
    t_ins_pt = get_subtree(t, path(tx_pt)[:-2])
    s_ins_pt = get_subtree(s, path(sx_pt)[:-2])
    
    # Perform the crossover: it's just a swap
    tx_st = tx_pt[1]
    sx_st = sx_pt[1]
    t_idx = path(tx_pt)[-2]
    s_idx = path(sx_pt)[-2]
    t_ins_pt[t_idx], s_ins_pt[s_idx] = sx_pt[1], tx_pt[1]

def main():
    from ponyge import Grammar
    grammar = Grammar("grammars/letter.bnf")
    
    dt = random_dt(grammar)
    print "dt", dt
    print "dt derivation", derived_str(dt)

    print "subtree (1,)", get_subtree(dt, (1,))
    print "node (1,)", get_node(dt, (1,))
    print "subtree (1, 1)", get_subtree(dt, (1, 1))
    print "node (1, 1)", get_node(dt, (1, 1))

    ds = random_dt(grammar)
    print "ds", ds
    print "ds derivation", derived_str(ds)

    
    dt_crossover(dt, ds, grammar)
    print derived_str(dt)

    dt_mutation(dt, grammar)
    print derived_str(dt)

    print "++++++++++++++++++"
    for item in traverse(dt):
        print item

if __name__ == "__main__":
    main()
