#!/usr/bin/env python

import sys
import random
import operator
from itertools import product
import numpy as np
from numpy import add, subtract, multiply, divide, sin, cos, exp, log, power, square
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

class BooleanProblemGeneral:
    """A compound problem. It consists of a single Boolean problem, at
    several sizes. Some sizes are used for training, some for
    testing. Training fitness is the sum of fitness on the training
    sub-problems. Testing fitness is the sum of fitness on the testing
    sub-problems."""
    def __init__(self, train_ns, test_ns, target):
        self.maximise = False
        self.train_problems = [
            BooleanProblem(n, target) for n in train_ns]
        self.test_problems = [
            BooleanProblem(n, target) for n in test_ns]
    def __call__(self, s):
        return sum([p(s) for p in self.train_problems])
    def test(self, s):
        return sum([p(s) for p in self.test_problems])
    
class SymbolicRegressionFitnessFunction:
    """Fitness function for symbolic regression problems. Yes, it's a
    Verb in the Kingdom of Nouns
    (http://steve-yegge.blogspot.com/2006/03/execution-in-kingdom-of-nouns.html).
    The reason is that the function needs some extra data to go with
    it: an arity, a list of bounds and increments to create the test
    cases. TODO could use correlation coefficient instead of RMSE. TODO
    Make a field to indicate what error metric we are using. TODO
    allow reading fitness cases, either just x-values or both x and
    y-values, from a file. TODO could report hits."""

    def __init__(self, target, train, test1=None, test2=None, defn="rmse"):
        """Pass in a target function and parameters for building the
        fitness cases for input variables for training and for testing
        if necessary. The cases are specified as a dictionary
        containing bounds and other variables: either a regular mesh
        over the ranges or randomly-generated points within the ranges
        can be generated. We allow one set of cases for training, and
        zero, one or two for testing, because several of our benchmark
        functions need two discontinuous ranges for testing data. If
        no testing cases are specified, the training data is used for
        testing as well. Can pass a keyword indicating the fitness
        definition, which can be 'rmse' (minimise root mean square
        error), 'correlation' (minimise 1-r**2), or 'hits' (maximise
        number of fitness cases within a small threshold). """

        # Training data
        self.cases = self.build_column_mesh_np(self.build_cases(**train))
        self.values = target(self.cases)

        # Testing data -- FIXME this could be neater.
        if test1 and test2:
            self.testing_cases = self.build_column_mesh_np(self.build_cases(**test1) + 
                                                           self.build_cases(**test2))
            self.testing_values = target(self.testing_cases)
        elif test1:
            self.testing_cases = self.build_column_mesh_np(self.build_cases(**test1))
            self.testing_values = target(self.testing_cases)
        else:
            # No special testing cases -- use training cases
            self.testing_cases = self.cases
            self.testing_values = self.values

        self.arity = target.func_code.co_argcount
        if defn == "rmse":
            self.maximise = False
            self.defn = self.rmse
        elif defn == "correlation":
            self.maximise = False
            self.defn = self.correlation_fitness
        elif defn == "hits":
            self.maximise = True
            self.defn = self.hits_fitness
        else:
            raise ValueError("Bad value for fitness definition: " + defn)
            

    def __call__(self, s):
        """Allow objects of this type to be called as if they were
        functions. Return a fitness value."""
        # s is a string which evals to a fn.
        try:
            fn = eval(s)
        except MemoryError:
            # can happen when the individual is too large to parse
            return default_fitness(self.maximise)
        try:
            v = self.defn(self.values, fn(self.cases))
            return v
        except FloatingPointError as fpe:
            return default_fitness(self.maximise)
        except ValueError as ve:
            print("ValueError: " + str(ve) +':' + s)
            raise
        except TypeError as te:
            print("TypeError: " + str(te) +':' + s)
            raise

    def test(self, s):
        """Test ind on unseen data. Return a fitness value."""
        # s is a string which evals to a fn.
        fn = eval(s)
        try:
            return self.defn(self.testing_values, fn(self.testing_cases))
        except FloatingPointError:
            return default_fitness(self.maximise)

    @staticmethod
    def build_cases(minv, maxv, incrv=None, randomx=None, ncases=None):
        """Generate fitness cases, either randomly or in a mesh."""
        if randomx is True:
            # incrv is ignored
            return SymbolicRegressionFitnessFunction.build_random_cases(minv, maxv, ncases)
        else:
            # ncases is ignored
            return SymbolicRegressionFitnessFunction.build_mesh(minv, maxv, incrv)

    @staticmethod
    def rmse(x, y):
        """Calculate root mean square error between x and y, two numpy arrays."""
        m = x - y
        m = square(m)
        m = np.mean(m)
        m = np.sqrt(m)
        return m

    @staticmethod
    def hits_fitness(x, y):
        """Hits as fitness: how many of the errors are very small?
        Minimise 1 - the proportion.

        """
        errors = abs(x - y)
        return 1 - np.mean(errors < 0.01)

    @staticmethod
    def correlation_fitness(x, y):
        """Correlation coefficient as fitness: minimise 1 - R^2."""
        try:
            # use [0][1] to get the right element from corr matrix
            corr = abs(np.corrcoef(x, y)[0][1])
        except (ValueError, FloatingPointError):
            # ValueError raised when x is a scalar because individual does
            # not depend on input. FloatingPointError raised when elements
            # of x are all identical.
            corr = 0.0
        return 1.0 - corr * corr

    @staticmethod
    def build_random_cases(minv, maxv, n):
        """Create a list of n lists, each list being x-coordinates for a
        fitness case. Generate them randomly within the bounds given by
        minv and maxv."""
        return [[random.uniform(lb, ub) for lb, ub in zip(minv, maxv)]
                for i in range(n)]

    @staticmethod
    def build_mesh(minv, maxv, increment):
        """Build a mesh, i.e. enumerate all points within the volume
        specified by the minv and maxv lists, at increment distances
        apart. Uses itertools.product.

        Two constraints on the input parameters: The three lists provided
        as parameters should be of the same length; and minv[i] <= maxv[i]
        for all i.

        Thanks to David White for an original Java implementation of this
        code, and comments.

        @param minv A list of minimum values for the n variables.
        @param maxv A list of maximum values for the n variables.
        @param increment A list of increments for the n variables.
        @return A list of tuples. Each tuple is the coordinates of a
        particular point."""

        assert len(minv) == len(maxv) == len(increment)
        one_d_meshes = []
        for minvi, maxvi, inci in zip(minv, maxv, increment):
            assert minvi <= maxvi
            nsteps = int((maxvi - minvi) / float(inci)) # eg [0, 10, 1] gives 10
            # note +1 to reach max
            mesh = [minvi + inci * i for i in range(nsteps + 1)]
            one_d_meshes.append(mesh)
        p = list(product(*one_d_meshes))
        return p

    @staticmethod
    def test_build_random():
        """Test -- this should print 100 points in correct ranges."""
        minv = [0.0, 0.0]
        maxv = [2.0, 2.0]
        n = 100
        mesh = SymbolicRegressionFitnessFunction.build_random_cases(minv, maxv, n)
        print(len(mesh))
        for item in mesh:
            print(item)

    @staticmethod
    def test_build_mesh():
        """Test -- this should print 63 points:
        [0.0, 0.0]
        [0.0, 1.0]
        [0.0, 2.0]
        [...]
        [2.0000000000000004, 0.0]
        [2.0000000000000004, 1.0]
        [2.0000000000000004, 2.0]"""
        minv = [0.0, 0.0]
        maxv = [2.0, 2.0]
        incrv = [0.1, 1.0]
        mesh = SymbolicRegressionFitnessFunction.build_mesh(minv, maxv, incrv)
        print(len(mesh))
        for item in mesh:
            print(item)

    @staticmethod
    def build_column_mesh_np(in_mesh):
        """Given a mesh of fitness cases, build a column-wise mesh
        consisting of multiple numpy arrays. Each array represents the
        values of an input variable."""
        mesh = np.array(in_mesh)
        mesh = mesh.transpose()
        return mesh

def benchmarks():
    return {
        "identity": SymbolicRegressionFitnessFunction(
            lambda x: x,
            {"minv": [0.0], "maxv": [1.0], "incrv": [0.1]}),
        
        "vladislavleva_12": SymbolicRegressionFitnessFunction(
            lambda x: exp(-x[0]) * power(x[0], 3.0) * cos(x[0]) * sin(x[0]) \
                * ((cos(x[0]) * power(sin(x[0]), 2.0)) - 1.0),
            {"minv": [0.05], "maxv": [10.0], "incrv": [0.1]}),

        "pagie_2d": SymbolicRegressionFitnessFunction(
            lambda x: (1 / (1 + x[0] ** -4) + 1 / (1 + x[1] ** -4)),
            {"minv": [-5, -5], "maxv": [5, 5], "incrv": [0.4, 0.4]}),

        "pagie_3d": SymbolicRegressionFitnessFunction(
            lambda x: (1 / (1 + x[0] ** -4) + 1 / (1 + x[1] ** -4)
                       + 1 / (1 + x[2] ** -4)),
            {"minv": [-5, -5, -5], "maxv": [5, 5, 5], "incrv": [0.4, 0.4, 0.4]}),

        }
   
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

    elif sys.argv[1] == "test_sr":
        sr = benchmarks()["vladislavleva_12"]
        g = "lambda x: 2*x"
        print(sr(g))
        sr = benchmarks()["identity"]
        g = "lambda x: x"
        print(sr(g))
    elif sys.argv[1] == "test_sr_mesh":
        test_build_mesh()
    elif sys.argv[1] == "test_sr_random":
        test_build_random()
