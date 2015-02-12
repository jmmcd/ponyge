#!/usr/bin/env python

from itertools import product
import random
import numpy as np
from numpy import sin, cos, exp, log, power, square
import os.path
import sys

# This is a standalone program which generates tables of data in a
# format suitable for use in typical regression software, including in
# ponyge.py's SymbolicRegressionFitnessFunction.
#
# The data comes from a known numerical function, eg the notorious
# quartic polynomial x + x**2 + x**3 + x**4 ("I've often dreamed of
# writing a paper "Death to Quartic Polynomial" because it's such a
# waste of time, giving worse-than-zero information on GP performance
# for anything real").
#
# The format is X[0] in the first column, X[1] in the next, up to
# X[n-1], and y in the last column, separated by spaces.
#
# The motivation here is the experimental work by Nicolau and Agapitos
# (forthcoming) who found that when using a known function as a
# target, eg a quartic polynomial with *randomly drawn* test cases, we
# can end up with a very easy or a very difficult test set depending
# on the randomness. That is dangerous. Therefore the correct policy
# is to use a standalone program like this one in advance, setting the
# seed to 0, to generate the training and test sets. These sets must
# then be published. The regression software must use these sets,
# rather than running the known function to generate its training and
# test sets. The benefit is replicability and reliability of
# results. A beneficial side-effect is that the regression software
# only needs to know how to read data files in a standard format
# rather than concerning itself with details of running functions to
# generate data.

def generate(fn, train, test1=None, test2=None):
    """Pass in a target function and parameters for building the fitness
    cases for input variables for training and for testing if
    necessary. The cases are specified as a dictionary containing
    bounds and other variables: either a regular mesh over the ranges
    or randomly-generated points within the ranges can be
    generated. We allow one set of cases for training, and zero, one
    or two for testing, because several of our benchmark functions
    need two discontinuous ranges for testing data.
    """

    train_Xy = test_Xy = None

    # Training data
    train_X = build_cases(**train)
    train_y = fn(train_X)
    train_Xy = np.vstack((train_X, train_y)).T

    # Testing data
    if test1:
        test_X = build_cases(**test1)
        if test2:
            test_X = np.hstack((test_X, build_cases(**test2)))
        test_y = fn(test_X)
        test_Xy = np.vstack((test_X, test_y)).T
    return train_Xy, test_Xy

def write(train_Xy, test_Xy, basename, delimiter=" "):
    """Write out the data. If necessary make a directory."""
    if not os.path.exists("data"):
        os.makedirs("data")
    filename = os.path.join("data", basename + "_train.dat")
    np.savetxt(filename, train_Xy, delimiter=delimiter)
    print("Wrote training data of shape: " + str(train_Xy.shape)
          + " to: " + filename)

    if test_Xy is not None:
        filename = os.path.join("data", basename + "_test.dat")
        np.savetxt(filename, test_Xy, delimiter=delimiter)
        print("Wrote test data of shape    : " + str(test_Xy.shape)
              + " to: " + filename)

def build_cases(minv, maxv, incrv=None, randomx=None, ncases=None):
    """Generate fitness cases, either randomly or in a mesh."""
    if randomx is True:
        # incrv is ignored, ncases is required
        X = np.array([np.random.uniform(mn, mx, ncases)
                      for mn, mx in zip(minv, maxv)])
    else:
        if incrv:
            # use incrv. create ncases as a list as below.
            ncases = [1 + (mx - mn) / float(inc)
                      for mn, mx, inc in zip(minv, maxv, incrv)]

        # ncases must be a list of the number of grid points in each dimension
        if len(minv) == 1:
            X = np.array([np.linspace(minv[0], maxv[0], ncases[0])])
        else:
            xs = [np.linspace(mn, mx, ncs) for mn, mx, ncs in zip(minv, maxv, ncases)]
            X = np.array(list(product(*xs))).T
    return X

def benchmarks(key):
    if   key == "identity":
        return generate(
            lambda x: x,
            {"minv": [0.0], "maxv": [1.0], "incrv": [0.1]})

    elif key == "vladislavleva-12":
        return generate(
            lambda x: exp(-x[0]) * power(x[0], 3.0) * cos(x[0]) * sin(x[0]) \
                * ((cos(x[0]) * power(sin(x[0]), 2.0)) - 1.0),
            {"minv": [0.05], "maxv": [10.0], "incrv": [0.1]},
            test1={"minv": [-0.5], "maxv": [10.5], "incrv": [0.05]})

    elif key == "vladislavleva-14":
        return generate(
            lambda x: 10.0 / (5.0 + sum((x[i]-3)**2 for i in range(5))),
            {"minv": [0.05]*5, "maxv": [6.05]*5, "randomx": True, "ncases": 1024},
            test1={"minv": [-0.25]*5, "maxv": [6.35]*5, "randomx": True, "ncases": 5000})

    elif key == "korns-12":
        # note we have 5D data even though the fn uses only the first 2D
        return generate(
            lambda x: 2.0 - (2.1 * (cos (9.8*x[0])*sin(1.3*x[1]))),
            {"minv": [-50,-50,-50,-50,-50],
             "maxv": [50,50,50,50,50], "randomx": True, "ncases": 10000},
            test1={"minv": [-50,-50,-50,-50,-50],
                   "maxv": [50,50,50,50,50,50], "randomx": True, "ncases": 10000})

    elif key == "pagie-2d":
        return generate(
            lambda x: (1 / (1 + x[0] ** -4) + 1 / (1 + x[1] ** -4)),
            {"minv": [-5, -5], "maxv": [5, 5], "incrv": [0.4, 0.4]})

    elif key == "pagie-3d":
        return generate(
            lambda x: (1 / (1 + x[0] ** -4) + 1 / (1 + x[1] ** -4)
                       + 1 / (1 + x[2] ** -4)),
            {"minv": [-5, -5, -5], "maxv": [5, 5, 5], "incrv": [0.4, 0.4, 0.4]})

    elif key == "nguyen-7":
        return generate(
            lambda x: log(x[0] + 1) + log(x[0]**2 + 1),
            {"minv": [0], "maxv": [2], "randomx": True, "ncases": 20})

    elif key == "fagan":
        return generate(
            lambda x: x[0]**4 + x[1]**2,
            {"minv": [-3, -3], "maxv": [3, 3], "incrv": [0.4, 0.4]},
            test1={"minv": [-5, -5], "maxv": [-3, -3], "incrv": [0.4, 0.4]},
            test2={"minv": [3, 3], "maxv": [5, 5], "incrv": [0.4, 0.4]})

    # FIXME not sure how to implement this -- x[0] is a column,
    # so we can't take range(x[0])...
    # elif key == "keijzer-6":
    #     return generate(
    #         lambda x: sum(1.0/i for i in range(x[0])),
    #         {"minv": [1], "maxv": [50], "incrv": [1]},
    #         test1={"minv": [1], "maxv": [120], "incrv": [1]})


    else:
        print "Unknown benchmark problem " + key


# Usage:
#
# generate_sr_data.py nguyen-y
# will write out nguyen-7_train.dat
#
# generate_sr_data.py vladislavleva-12
# will write out vladislavleva-12_train.dat and vladislavleva-12_test.dat

key = sys.argv[1]
if len(sys.argv) > 2:
    delimiter = sys.argv[2]
else:
    delimiter = " "
random.seed(0)
train_Xy, test_Xy = benchmarks(key)
write(train_Xy, test_Xy, key, delimiter=delimiter)
