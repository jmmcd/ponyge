#!/usr/bin/env python

import sys
import random
import operator
from itertools import product
import numpy as np
from numpy import logical_and, logical_or, logical_xor, logical_not
np.seterr(all='raise')

def eval_or_exec(expr):
    """Use eval or exec to interpret expr.

    A limitation in Python is the distinction between eval and
    exec. The former can only be used to return the value of a simple
    expression (not a statement) and the latter does not return
    anything."""

    #print(s)
    try:
        retval = eval(expr)
    except SyntaxError:
        # SyntaxError will be thrown by eval() if s is compound,
        # ie not a simple expression, eg if it contains function
        # definitions, multiple lines, etc. Then we must use
        # exec(). Then we assume that s will define a variable
        # called "XXXeval_or_exec_outputXXX", and we'll use that.
        dictionary = {}
        exec(expr, dictionary)
        retval = dictionary["XXXeval_or_exec_outputXXX"]
    except MemoryError:
        # Will be thrown by eval(s) or exec(s) if s contains over-deep
        # nesting (see http://bugs.python.org/issue3971). The amount
        # of nesting allowed varies between versions, is quite low in
        # Python2.5. If we can't evaluate, phenotype is marked
        # invalid.
        retval = None
    return retval

def default_fitness(maximise):
    """Return default fitness given maximization of minimization"""
    if maximise:
        return -100000.0
    else:
        return 100000.0

class RandomFitness:
    """Useful for investigating algorithm dynamics in the absence of
    selection pressure. Fitness is random."""

    def __call__(self, ind):
        """Allow objects of this type to be called as if they were
        functions. Return a random value as fitness."""
        return random.random()

class SizeFitness:
    """Useful for investigating control of tree size. Return the
    difference from a target size."""
    maximise = False
    def __init__(self, target_size=20):
        self.target_size = target_size

    def __call__(self, ind):
        """Allow objects of this type to be called as if they were
        functions."""
        return abs(self.target_size - len(ind))

class MaxFitness():
    """Arithmetic maximisation with python evaluation."""
    maximise = True
    def __call__(self, candidate):
        return eval_or_exec(candidate)


class BooleanProblem:
    """Boolean problem of size n. Pass target function in.
    Minimises. Objects of this type can be called."""

    # TODO could benchmark this versus sub-machine code
    # implementation.
    def __init__(self, n, target):
        self.maximise = False
        
        # make all possible fitness cases
        vals = [False, True]
        p = list(product(*[vals for i in range(n)]))
        self.x = np.transpose(p)

        # get target function's values on fitness cases
        try:
            # assume target is a function
            self.target_cases = target(self.x)
        except TypeError:
            # no, target was a list of values
            if len(target) != 2 ** n:
                s = "Wrong number of target cases (%d) for problem size %d" % (
                    len(target), n)
                raise ValueError(s)
            self.target_cases = np.array(target)

    def __call__(self, s):
        # s is a string which evals to a fn.
        fn = eval(s)
        output = fn(self.x)
        non_matches = output ^ self.target_cases
        return sum(non_matches) # Fitness is number of errors


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: fitness.py <keyword>.")
        sys.exit()

    elif sys.argv[1] == "test_boolean":
        fn1 = "lambda x: ~(x[0] ^ x[1] ^ x[2] ^ x[3] ^ x[4])" # even-5 parity
        fn2 = "lambda x: True" # matches e5p for 16 cases out of 32
        fn3 = "lambda x: x[0] ^ x[1] ^ x[2] ^ x[3] ^ x[4]" # never matches e5p

        b = BooleanProblem(5, eval(fn1)) # target is e5p itself
        print(b(fn1)) # fitness = 0
        print(b(fn2)) # fitness = 16
        print(b(fn3)) # fitness = 32

        # Can also pass in a list of target values, ie semantics phenotype
        b = BooleanProblem(2, [False, False, False, False])
        print(b(fn2))

    elif sys.argv[1] == "test_random":
        fn1 = "dummy"
        r = RandomFitness()
        print(r(fn1))

    elif sys.argv[1] == "test_size":
        fn1 = "0123456789"
        fn2 = "01234567890123456789"
        s = SizeFitness(20)
        print(s(fn1))
        print(s(fn2))

    elif sys.argv[1] == "test_max":
        fn1 = "((0.5 + 0.5) + (0.5 + 0.5)) * ((0.5 + 0.5) + (0.5 + 0.5))"
        m = MaxFitness()
        print(m(fn1))
