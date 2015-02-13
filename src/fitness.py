#!/usr/bin/env python

import sys
import random
import operator
import subprocess
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

def run_cmd(cmd):
    """executes a command line command"""
    process = subprocess.Popen(cmd, shell=True,
                               stdout=subprocess.PIPE,
                               stdin=subprocess.PIPE)
    result = process.communicate()
    return result

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

class GretlFitness():
    """GRETL is a GNU econometrics library, this outputs gretl scripts
    and then parses the output and significance to generate a fitness  value"""
    maximise = True
    def __call__(self, candidate):
        candidate = candidate.replace('\\n', '\n')
        outfile = open('gretl.inp','wb')
        outfile.write(candidate)
        outfile.close()
        result = run_cmd("gretlcli -b gretl.inp 2> /dev/null")
        fitness = self.parse_result(result)
        return fitness

    def parse_result(self, result):
        print "evaluating"
        fitness = 0
        lines = result[0].split('\n')
        for line in lines:
            #            # coefficient   std. error   t-ratio   p-value
            # if '*' in line:
            #     print line
            if 'R-squared' in line:
                line = line.split(' ')
                line = filter(lambda a: a != '', line)
                fitness = line[1]
        return fitness

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
    The function needs some extra data to go with it -- it stores
    train and test data etc."""

    @classmethod
    def __init__(self, filename, test_filename=None,
                 split=0.9,
                 randomise=False, defn="rmse"):
        """Construct an SRFF by reading training data from a file. If
        a test_filename is given, get test data there. Else split the
        data according to split, eg 0.9 means 90% for training, 10%
        for testing. If randomise is True, take random rows; else take
        the last rows as test data. Tries to handle csv files
        correctly, else genfromtxt assumes it is whitespace-separated.
        # character is used for comment lines."""

        if filename.endswith(".csv"):
            delimiter = ","
        else:
            delimiter = None
        d = np.genfromtxt(filename, delimiter=delimiter)
        if randomise:
            # shuffle the rows before allocating train/test
            np.random.shuffle(d)
        dX = d[:,:-1]
        dy = d[:,-1]
        if test_filename:
            # separate filename for test
            train_X = dX.T
            train_y = dy
            # assume same filename suffix and delimiter
            d = np.genfromtxt(test_filename, delimiter=delimiter)
            test_X = d[:,:-1].T
            test_y = d[:,-1]
        elif split:
            # get data from same file for both train and test
            idx = int(split * len(d))
            train_X = dX[:idx].T
            train_y = dy[:idx]
            test_X = dX[idx:].T
            test_y = dy[idx:]
        else:
            # same data for train and test
            test_X = train_X = dX.T
            test_y = train_y = dy

        # print "len(train_X), len(test_X)"
        # print len(train_X), len(test_X)
        # print "len(train_X[0]), len(train_y), len(test_X[0]), len(test_y)"
        # print len(train_X[0]), len(train_y), len(test_X[0]), len(test_y)
        assert len(train_X.shape) == len(test_X.shape) == 2
        assert len(train_y.shape) == len(test_y.shape) == 1
        assert len(train_X) == len(test_X) # how many independent vars
        assert len(train_X[0]) == len(train_y)
        assert len(test_X[0]) == len(test_y)

        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.arity = len(train_X)
        if defn == "rmse":
            self.maximise = False
            self.defn = self.rmse
        elif defn == "correlation":
            self.maximise = False
            self.defn = self.correlation_fitness
        elif defn == "log_error":
            self.maximise = False
            self.defn = self.log_error
        elif defn == "hits":
            self.maximise = True
            self.defn = self.hits_fitness
        elif defn == "classification_accuracy":
            self.maximise = True
            self.defn = self.class_acc_fitness
        else:
            raise ValueError("Bad value for fitness definition: " + defn)


    def __call__(self, fn):
        """Allow objects of this type to be called as if they were
        functions. Return just a fitness value."""
        return self.get_semantics(fn)[0]

    def get_semantics(self, fn, test=False):
        """Run the function over the training set. Return the fitness
        and the vector of results (the "semantics" of the function).
        Return (default_fitness, None) on error. Pass test=True to run
        the function over the testing set instead."""

        # print("s:", s)
        if not callable(fn):
            # assume fn is a string which evals to a function.
            try:
                fn = eval(fn)
            except MemoryError:
                # print("MemoryError in get_semantics()")
                # self.memo[s, test] = default_fitness(self.maximise), None
                # return self.memo[s, test]
                return default_fitness(self.maximise), None

        try:
            if not test:
                vals_at_cases = fn(self.train_X)
                fit = self.defn(self.train_y, vals_at_cases)
            else:
                vals_at_cases = fn(self.test_X)
                fit = self.defn(self.test_y, vals_at_cases)

            return fit, vals_at_cases
            # self.memo[s, test] = fit, vals_at_cases
            # return self.memo[s, test]

        except FloatingPointError as fpe:
            # print("FloatingPointError in get_semantics()")
            return default_fitness(self.maximise), None
            # self.memo[s, test] = default_fitness(self.maximise), None
            # return self.memo[s, test]
        except ValueError as ve:
            print("ValueError: " + str(ve) +':' + str(fn))
            raise
        except TypeError as te:
            print("TypeError: " + str(te) +':' + str(fn))
            raise

    def test(self, fn):
        """Test ind on unseen data. Return a fitness value."""
        return self.get_semantics(fn, True)[0]

    @staticmethod
    def rmse(self, y, yhat):
        """Calculate root mean square error between yhat and y, two numpy
        arrays."""
        m = yhat - y
        m = np.square(m)
        m = np.mean(m)
        m = np.sqrt(m)
        return m

    @staticmethod
    def log_error(self, y, yhat):
        """Calculate log error between yhat and y, two numpy arrays."""
        return 1.0 * np.mean(np.log(1.0+(np.abs(yhat-y))))

    @staticmethod
    def hits_fitness(self, y, yhat):
        """Hits as fitness: how many of the errors are very small?
        Minimise 1 - the proportion."""
        errors = abs(yhat - y)
        return 1 - np.mean(errors < 0.01)

    @staticmethod
    def class_acc_fitness(self, y, yhat):
        """Classification accuracy fitness: how many of the
        predictions are of the same sign as the correct value?
        Maximise."""
        signy = np.sign(y)
        signyhat = np.sign(yhat)
        tp_plus_tn = np.count_nonzero((signy * signyhat) >= 0)
        return tp_plus_tn / float(len(y))

    @staticmethod
    def correlation_fitness(self, y, yhat):
        """Correlation coefficient as fitness: minimise 1 - R^2."""
        try:
            # use [0][1] to get the right element from corr matrix
            corr = abs(np.corrcoef(yhat, y)[0][1])
        except (ValueError, FloatingPointError):
            # ValueError raised when yhat is a scalar because individual does
            # not depend on input. FloatingPointError raised when elements
            # of yhat are all identical.
            corr = 0.0
        return 1.0 - corr * corr


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
