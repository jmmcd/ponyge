#!/usr/bin/env python

import multiprocessing
import numpy as np
import os
from pylab import *

reps = 30
cores = multiprocessing.cpu_count()

def process_dir(dirname, basefilename, reps):
    best, codons = [], []
    for i in range(reps):
        filename = os.path.join(dirname, basefilename + str(i) + ".dat")
        data = open(filename).read()
        best_i, codons_i = [], []
        for line in data.split("\n"):
            if (not line.startswith("#")) and (not line.startswith("Best")):
                numbers, phenotype = line.split(":", maxsplit=1)
                (gen, evals, bestfit, bestcodons, meanfit, stdfit,
                 meancodons, stdcodons, ninvalids) = map(float, numbers.split())
                best_i.append(bestfit)
                codons_i.append(bestcodons)
        best.append(best_i)
        codons.append(codons_i)
    make_figure(best, dirname, basefilename, "best")
    make_figure(codons, dirname, basefilename, "codons")

def make_figure(d, dirname, basefilename, key):
    d = np.array(d)
    d = d.transpose()
    d_mean = np.mean(d, 1) # new array of per-generation mean
    d_std = np.std(d, 1) # new array of per-generation stddev
    fig = figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.errorbar(range(len(d_mean)), d_mean, yerr=d_std, fmt='-o')
    ax.grid(True)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    filename = os.path.join(dirname, basefilename + key + ".pdf")
    fig.savefig(filename)
    close()

def graph(basedir):
    for problem in ["sr", "bool"]: # later do "bool" also
        for grammar in ["bnf"]: # later do ebnf also
            for cond in ["int", "dt"]:
                dir = os.path.join(
                    basedir,
                    problem,
                    "bnf_ebnf_int_dt")
                process_dir(dir, grammar + "_" + cond + "_", reps)
                
def run(basedir):
    proc_idx = 0
    for problem in ["sr"]:
        if problem == "sr":
            fitness_arg = """ -f 'fitness.benchmarks()["pagie_2d"]' """
            grammar_arg = """ -b 'grammars/symbolic_regression_2d.bnf' """
        elif problem == "bool":
            fitness_arg = """ -f 'fitness.BooleanProblem(5, lambda x: ~(x[0] ^ x[1] ^ x[2] ^ x[3] ^ x[4]))' """
            grammar_arg = """ -b 'grammars/boolean.bnf' """
        else:
            print("Unknown problem " + problem)
            sys.exit(1)
            
        for grammar in ["bnf"]: # later do ebnf also
            for cond in ["int"]: # later do dt also
                for rep in range(reps):
                    dir = os.path.join(
                        basedir,
                        problem)
                    filename = os.path.join(
                        dir, grammar + "_" + cond + "_" + str(rep) + ".dat")
                    cmd = ("python ponyge.py -p 1000 "
                           + ("-d" if cond == "dt" else "")
                           + fitness_arg + grammar_arg
                           + " > " + filename)
                    # simple hack to use multiple cores
                    if (proc_idx % cores) != (cores - 1):
                        cmd += " &" 
                    proc_idx += 1
                    print(cmd)
                    # os.system(cmd)

if __name__ == "__main__":
    if len(sys.argv) == 3:
        if sys.argv[1] == "run":
            run(sys.argv[2])
        elif sys.argv[1] == "graph":
            graph(sys.argv[2])
