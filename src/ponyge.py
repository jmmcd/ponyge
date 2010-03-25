#! /usr/bin/env python

# PonyGE
# Copyright (c) 2009 Erik Hemberg and James McDermott
# Hereby licensed under the GNU GPL v3.

import sys, copy, re, random, math

class Grammar(object):
    NT = "NT" # Non Terminal
    T = "T" # Terminal

    def __init__(self, file_name):
        if file_name.endswith("pybnf"):
            self.python_mode = True
        else:
            self.python_mode = False
        self.readBNFFile(file_name)

    def readBNFFile(self, file_name):
        """Read a grammar file in BNF format"""
        # <.+?> Non greedy match of anything between brackets
        NON_TERMINAL_PATTERN = "(<.+?>)"
        RULE_SEPARATOR = "::="
        PRODUCTION_SEPARATOR = "|"

        self.rules = {}
        self.non_terminals, self.terminals = set(), set()
        self.start_rule = None
        # Read the grammar file
        for line in open(file_name, 'r'):
            if not line.startswith("#") and line.strip() != "":
                # Split rules. Everything must be on one line
                #TODO Avoid everything on one line
                if line.find(RULE_SEPARATOR):
                    lhs, productions = line.split(RULE_SEPARATOR)
                    lhs = lhs.strip()
                    if not re.search(NON_TERMINAL_PATTERN, lhs):
                        #TODO correct error type?
                        raise ValueError("lhs is not a NT:",lhs)
                    self.non_terminals.add(lhs)
                    if self.start_rule == None:
                        self.start_rule = (lhs, self.NT)
                    # Find terminals
                    tmp_productions = []
                    for production in [production.strip()
                                       for production in productions.split(PRODUCTION_SEPARATOR)]:
                        tmp_production = []
                        if not re.search(NON_TERMINAL_PATTERN, production):
                            self.terminals.add(production)
                            tmp_production.append((production, self.T))
                        else:
                            # Match non terminal or terminal pattern
                            # TODO does this handle quoted NT symbols
                            for value in re.findall("<.+?>|[^<>]*", production):
                                if value != '':
                                    if not re.search(NON_TERMINAL_PATTERN, value):
                                        symbol = (value, self.T)
                                    else:
                                        symbol = (value, self.NT)
                                    tmp_production.append(symbol)
                        tmp_productions.append(tmp_production)
                    # Create a rule
                    if not lhs in self.rules:
                        self.rules[lhs] = tmp_productions
                    else:
                        raise ValueError("lhs should be unique", lhs)
                else:
                    raise ValueError("Each rule must be on one line")

    def generate(self, input, max_wraps=2):
        """Map input via rules to output. Returns output and used_input"""
        used_input = 0
        wraps = 0
        output = []

        unexpanded_symbols = [self.start_rule]
        while (wraps < max_wraps) and (len(unexpanded_symbols) > 0):
            # Wrap
            if used_input % len(input) == 0 and used_input > 0:
                wraps += 1
            # Expand a production
            current_symbol = unexpanded_symbols.pop(0)
            # Set output if it is a terminal
            if current_symbol[1] != self.NT:
                output.append(current_symbol[0])
            else:
                production_choices = self.rules[current_symbol[0]]
                # Select a production
                current_production = input[used_input % len(input)] % len(production_choices)
                # Use an input if there was more then 1 choice
                if len(production_choices) > 1:
                    used_input += 1
                # Derviation order is left to right(depth-first)
                unexpanded_symbols = production_choices[current_production] + unexpanded_symbols

        #Not completly expanded
        if len(unexpanded_symbols) > 0:
            return (None, 0)

        output = "".join(output)
        if self.python_mode:
            output = self.python_filter(output)
        return (output, used_input)

    # Create correct python syntax. We use {: and :} as special open
    # and close brackets, because it's not possible to specify
    # indentation correctly in a BNF grammar without this type of
    # scheme.
    def python_filter(self, txt):
        indent_level = 0
        for i in range(len(txt) - 1):
            tok = txt[i:i+2]
            if tok == "{:":
                indent_level += 1
            elif tok == ":}":
                indent_level -= 1
            tabstr = "\n" + "  " * indent_level
            if tok == "{:" or tok == ":}":
                txt = txt.replace(tok, tabstr, 1)
        # Strip superfluous blank lines.
        txt = "\n".join([line for line in txt.split("\n")
                         if line.strip() != ""])
        return txt

# An unpleasant limitation in Python is the distinction between
# eval and exec. The former can only be used to return the value
# of a simple expression (not a statement) and the latter does not
# return anything.
def eval_or_exec(s):
    #print(s)
    try:
        retval = eval(s)
    except SyntaxError:
        # SyntaxError will be thrown by eval() if s is compound,
        # ie not a simple expression, eg if it contains function
        # definitions, multiple lines, etc. Then we must use
        # exec(). Then we assume that s will define a variable
        # called "XXXeval_or_exec_outputXXX", and we'll use that.
        exec(s)
        retval = XXXeval_or_exec_outputXXX
    return retval

class StringMatch():
    """Fitness function for matching a string. Takes a string and
    returns fitness. Penalises output that is not the same length as
    the target. Usage: StringMatch("golden") returns a *callable
    object*, ie the fitness function."""
    maximise = False
    def __init__(self, s):
        self.target = s
    def __call__(self, x):
        fitness = max(len(self.target), len(x))
        # Loops as long as the shorter of two strings
        for (t, o) in zip(self.target, x):
            if t == o:
                fitness -= 1
        return fitness

class XORFitness():
    """XOR fitness function with python evaluation."""
    maximise = True
    def __call__(self, candidate):
        def xor(x, y):
            return (x and not y) or (y and not x)
        f = eval(candidate)
        fitness = 0
        for x in [False, True]:
            for y in [False, True]:
                if f(x, y) == xor(x, y):
                    fitness += 1
        return fitness

class MaxFitness():
    """Maximisation with python evaluation."""
    maximise = True
    def __call__(self, candidate):
        return eval_or_exec(candidate)

class Individual(object):
    """A GE individual"""
    def __init__(self, genome, length=100):
        if genome == None:
            self.genome = [random.randint(0, CODON_SIZE)
                           for i in range(length)]
        else:
            self.genome = genome
        if FITNESS_FUNCTION.maximise:
            self.fitness = -100000.0
        else:
            self.fitness = 100000.0
        self.phenotype = None
        self.used_codons = 0

    def __lt__(self, other):
        if FITNESS_FUNCTION.maximise:
            return other.fitness < self.fitness
        else:
            return self.fitness < other.fitness

    def __str__(self):
        return ("Individual: " +
                str(self.phenotype) + "; " + str(self.fitness))

    def evaluate(self, fitness):
        self.fitness = fitness(self.phenotype)

def initialise_population(size=10):
    """Create a popultaion of size and return"""
    return [Individual(None) for cnt in range(size)]

def print_stats(generation, individuals):
    #TODO print to file
    def ave(values):
        return float(sum(values))/len(values)
    def std(values, ave):
        return math.sqrt(float(sum([(value-ave)**2 for value in values]))/len(values))

    #TODO is invalid fitness handled properly in stat
    ave_fit = ave([ind.fitness for ind in individuals])
    std_fit = std([ind.fitness for ind in individuals], ave_fit)
    #TODO is invalid length handeled properly in stats
    ave_used_codons = ave([i.used_codons for i in individuals])
    std_used_codons = std([i.used_codons for i in individuals], ave_used_codons)
    print("Gen:%d evals:%d ave:%.2f+-%.3f aveUsedC:%.2f+-%.3f %s" % (generation, (GENERATION_SIZE*generation), ave_fit, std_fit, ave_used_codons, std_used_codons, individuals[0]))
#    print_individuals(individuals)

def int_flip_mutation(individual):
    """Mutate the individual by randomly chosing a new int with
    probability p_mut. Works per-codon, hence no need for
    "within_used" option."""
    for i in range(len(individual.genome)):
        if random.random() < MUTATION_PROBABILITY:
            individual.genome[i] = random.randint(0,CODON_SIZE)
    return individual

# Two selection methods: tournament and truncation
def tournament_selection(population, tournament_size=3):
    """Given an entire population, draw <tournament_size> competitors
    randomly and return the best."""
    winners = []
    while len(winners) < GENERATION_SIZE:
        competitors = random.sample(population, tournament_size)
        competitors.sort()
        winners.append(competitors[0])
    return winners

def truncation_selection(population, proportion=0.5):
    """Given an entire population, return the best <proportion> of
    them."""
    population.sort(reverse=True)
    cutoff = int(len(population) * float(proportion))
    return population[0:cutoff]

def onepoint_crossover(p, q, within_used=True):
    """Given two individuals, create two children using one-point
    crossover and return them."""
    # Get the chromosomes
    pc, qc = p.genome, q.genome
    # Uniformly generate crossover points. If within_used==True,
    # points will be within the used section.
    if within_used:
        maxp, maxq = p.used_codons, q.used_codons
    else:
        maxp, maxq = len(pc), len(qc)
    pt_p, pt_q = random.randint(1, maxp), random.randint(1, maxq)
    # Make new chromosomes by crossover: these slices perform copies
    if random.random() < CROSSOVER_PROBABILITY:
        c = pc[:pt_p] + qc[pt_q:]
        d = qc[:pt_q] + pc[pt_p:]
    else:
        c, d = pc, qc
    # Put the new chromosomes into new individuals
    return [Individual(c), Individual(d)]

def evaluate_fitness(individuals, grammar, fitness_function):
    # Perform the mapping for each individual
    for ind in individuals:
        ind.phenotype, ind.used_codons = grammar.generate(ind.genome)
        if ind.phenotype != None:
            ind.evaluate(fitness_function)

def generational_replacement(new_pop, individuals):
    for ind in individuals[:ELITE_SIZE]:
        new_pop.append(copy.copy(ind))
    new_pop.sort()
    return new_pop[:GENERATION_SIZE]

def steady_state_replacement(new_pop, individuals):
    individuals.sort()
    individuals[-1] = max(new_pop + individuals[-1:])
    return individuals

def search_loop(max_generations, individuals, grammar, replacement, selection, fitness_function):
    """Loop over max generations"""
    #Evaluate initial population
    evaluate_fitness(individuals, grammar, fitness_function)
    best_ever = min(individuals)
    individuals.sort()
    print_stats(1,individuals)
    for generation in range(2,(max_generations+1)):
        #Select parents
        parents = selection(individuals)
        #Crossover parents and add to the new population
        new_pop = []
        while len(new_pop) < GENERATION_SIZE:
            new_pop.extend(onepoint_crossover(*random.sample(parents, 2)))
        #Mutate the new population
        new_pop = list(map(int_flip_mutation, new_pop))
        #Evaluate the fitness of the new population
        evaluate_fitness(new_pop, grammar, fitness_function)
        #Replace the sorted individuals with the new populations
        individuals = replacement(new_pop, individuals)
        print_stats(generation, individuals)
        best_ever = min(best_ever, min(individuals))
    return best_ever

# TODO can the functions be structured better? Make the selction sizes clearer
CODON_SIZE = 127
ELITE_SIZE = 1
POPULATION_SIZE = 100
GENERATION_SIZE = 100
GENERATIONS = 30
MUTATION_PROBABILITY = 0.1
CROSSOVER_PROBABILITY = 0.7
GRAMMAR_FILE = "grammars/letter.bnf"
FITNESS_FUNCTION = StringMatch("golden")
# GRAMMAR_FILE = "grammars/arithmetic.pybnf"
# FITNESS_FUNCTION = MaxFitness()
# GRAMMAR_FILE = "grammars/boolean.pybnf"
# FITNESS_FUNCTION = XORFitness()

# Run program
def mane():
    # Read grammar
    bnf_grammar = Grammar(GRAMMAR_FILE)
    # Create Individuals
    individuals = initialise_population(POPULATION_SIZE)
    # Loop
    best_ever = search_loop(GENERATIONS, individuals, bnf_grammar, generational_replacement, tournament_selection, FITNESS_FUNCTION)
    print("Best" + str(best_ever))

if __name__ == "__main__":
    mane()
