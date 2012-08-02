#!/usr/bin/env python

import random
import operator

def eval_or_exec(expr):
    """Use eval or exec to interpret expr.

    A limitation in Python is the distinction between eval and
    exec. The former can only be used to return the value of a simple
    expression (not a statement) and the latter does not return
    anything."""

    #print(s)
    try:
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
        # Python2.5. If we can't evaluate, award bad fitness.
        retval = fitness.default_fitness(FITNESS_FUNCTION.maximise)
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
    def __init__(self, target_size=20):
        self.target_size = target_size

    def __call__(self, ind):
        """Allow objects of this type to be called as if they were
        functions."""
        return abs(self.target_size - len(ind))

class XORFitness():
    """XOR fitness function with python evaluation."""
    maximise = True
    def __call__(self, candidate):
        function = eval(candidate)
        fitness = 0
        for x_0 in [False, True]:
            for x_1 in [False, True]:
                if function(x_0, x_1) == operator.xor(x_0, x_1):
                    fitness += 1
        return fitness

class EvenNParityFitness():
    """EvenNParity fitness function with python evaluation."""

    maximise = True

    def head(self, xes):
        """Takes a list and returns the head"""
        return xes[0]

    def tail(self, xes):
        """Takes a list and returns the list with its head removed."""
        return xes[1:]

    def nand(self, x_0, x_1):
        """ Boolean nand """
        return not operator.and_(x_0, x_1)

    def nor(self, x_0, x_1):
        """ Boolean nor """
        return not operator.or_(x_0, x_1)

    def _and(self, x_0, x_1):
        """ Boolean and """
        return operator.and_(x_0, x_1)

    def _or(self, x_0, x_1):
        """ Boolean or """
        return operator.or_(x_0, x_1)

    def __init__(self, size):
        """0 is False, 1 is True. Lookup tables for parity input and
        output. Size is the number of inputs"""
        self._output = [None] * 2**size
        self._input = []
        for i in range(len(self._output)):
            self._output[i] = (bin(i).count('1') % 2) == 0
            self._input.append([False] * size)
            for j, cnt in zip(bin(i)[2:][::-1], range(size)):
                if j == '1':
                    self._input[-1][cnt] = True

        self._input = tuple(self._input)
        self._output =  tuple(self._output)

    def __call__(self, candidate):
        """Compare the output generated from the input by the
        candidate with the output given by the input in the lookup
        table."""
        code = compile(candidate, '<string>', 'eval')
        function = eval(code, locals())
        fitness = 0
        for x_0, x_1 in zip(self._output, self._input):
            try:
                if function(x_1) is x_0:
                    fitness += 1
            except (TypeError, IndexError, MemoryError):
                fitness = default_fitness(self.maximise)
                break
        return fitness

class MaxFitness():
    """Arithmetic maximisation with python evaluation."""
    maximise = True
    def __call__(self, candidate):
        return eval_or_exec(candidate)
