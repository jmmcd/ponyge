#! /usr/bin/env python

# PonyGE
# Copyright (c) 2009-2012 Erik Hemberg and James McDermott
# Hereby licensed under the GNU GPL v3.
# http://ponyge.googlecode.com

"""Small GE implementation."""

import sys, copy, re, random, math, operator
import fitness
import derivation_tree as dt

class Grammar(object):
    """Context Free Grammar"""
    NT = "NT" # Non Terminal
    T = "T" # Terminal

    def __init__(self, file_name, nvars=None):
        if file_name.endswith("pybnf"):
            self.python_mode = True
        else:
            self.python_mode = False
        self.rules = {}
        self.non_terminals, self.terminals = set(), set()
        self.start_rule = None

        self.read_bnf_file(file_name, nvars)

    def read_bnf_file(self, file_name, nvars=None):
        """Read a grammar file in BNF format"""
        rule_separator = "::="
        # Don't allow space in NTs, and use lookbehind to match "<"
        # and ">" only if not preceded by backslash. Group the whole
        # thing with capturing parentheses so that split() will return
        # all NTs and Ts. TODO does this handle quoted NT symbols?
        non_terminal_pattern = r"((?<!\\)<\S+?(?<!\\)>)"
        # Use lookbehind again to match "|" only if not preceded by
        # backslash. Don't group, so split() will return only the
        # productions, not the separators.
        production_separator = r"(?<!\\)\|"

        # Read the grammar file
        for line in open(file_name, 'r'):
            if not line.startswith("#") and line.strip() != "":
                # Split rules. Everything must be on one line
                if line.find(rule_separator):
                    lhs, productions = line.split(rule_separator, 1) # 1 split
                    lhs = lhs.strip()
                    if not re.search(non_terminal_pattern, lhs):
                        raise ValueError("lhs is not a NT:", lhs)
                    self.non_terminals.add(lhs)
                    if self.start_rule == None:
                        self.start_rule = (lhs, self.NT)
                    if lhs == "<var>" and nvars is not None:
                        # Respond to nvars if we have it. Ignore any
                        # RHS in the file. Use x[0] | ... | x[n-1]
                        tmp_productions = []
                        for i in range(nvars):
                            tmp_production = []
                            tmp_production.append(("x[%d]"%i, self.T))
                            tmp_productions.append(tmp_production)
                        if not lhs in self.rules:
                            self.rules[lhs] = tmp_productions
                        else:
                            raise ValueError("lhs should be unique", lhs)
                        continue
                    # Find terminals and non-terminals
                    tmp_productions = []
                    for production in re.split(production_separator, productions):
                        production = production.strip().replace(r"\|", "|")
                        tmp_production = []
                        for symbol in re.split(non_terminal_pattern, production):
                            symbol = symbol.replace(r"\<", "<").replace(r"\>", ">")
                            if len(symbol) == 0:
                                continue
                            elif re.match(non_terminal_pattern, symbol):
                                tmp_production.append((symbol, self.NT))
                            else:
                                self.terminals.add(symbol)
                                tmp_production.append((symbol, self.T))

                        tmp_productions.append(tmp_production)
                    # Create a rule
                    if not lhs in self.rules:
                        self.rules[lhs] = tmp_productions
                    else:
                        raise ValueError("lhs should be unique", lhs)
                else:
                    raise ValueError("Each rule must be on one line")

    def __str__(self):
        return "%s %s %s %s" % (self.terminals, self.non_terminals,
                                self.rules, self.start_rule)

    def generate(self, _input, max_wraps=1):
        """Map input via rules to output. Returns output and used_input"""
        used_input = 0
        wraps = 0
        output = []
        production_choices = []

        unexpanded_symbols = [self.start_rule]
        while (wraps <= max_wraps) and (len(unexpanded_symbols) > 0):
            # Wrap
            if used_input % len(_input) == 0 and \
                    used_input > 0 and \
                    len(production_choices) > 1:
                wraps += 1
            # Expand a production
            current_symbol = unexpanded_symbols.pop(0)
            # Set output if it is a terminal
            if current_symbol[1] != self.NT:
                output.append(current_symbol[0])
            else:
                production_choices = self.rules[current_symbol[0]]
                # Select a production
                current_production = _input[used_input % len(_input)] % len(production_choices)
                # Use an input if there was more then 1 choice
                if len(production_choices) > 1:
                    used_input += 1
                # Derviation order is left to right(depth-first)
                unexpanded_symbols = production_choices[current_production] + unexpanded_symbols

        #Not completly expanded
        if len(unexpanded_symbols) > 0:
            return (None, used_input)

        output = "".join(output)
        if self.python_mode:
            output = python_filter(output)
        return (output, used_input)

def python_filter(txt):
    """Create correct python syntax.

    We use {: and :} as special open and close brackets, because
    it's not possible to specify indentation correctly in a BNF
    grammar without this type of scheme."""

    indent_level = 0
    tmp = txt[:]
    i = 0
    while i < len(tmp):
        tok = tmp[i:i+2]
        if tok == "{:":
            indent_level += 1
        elif tok == ":}":
            indent_level -= 1
        tabstr = "\n" + "  " * indent_level
        if tok == "{:" or tok == ":}":
            tmp = tmp.replace(tok, tabstr, 1)
        i += 1
    # Strip superfluous blank lines.
    txt = "\n".join([line for line in tmp.split("\n")
                     if line.strip() != ""])
    return txt

class StringMatch():
    """Fitness function for matching a string. Takes a string and
    returns fitness. Penalises output that is not the same length as
    the target. Usage: StringMatch("golden") returns a *callable
    object*, ie the fitness function."""
    maximise = False
    def __init__(self, target):
        self.target = target
    def __call__(self, guess):
        fitness = max(len(self.target), len(guess))
        # Loops as long as the shorter of two strings
        for (t_p, g_p) in zip(self.target, guess):
            if t_p == g_p:
                fitness -= 1
        return fitness

class Individual(object):
    """A GE individual"""
    def __init__(self, genome, length=100):
        if genome == None:
            self.genome = [random.randint(0, CODON_SIZE)
                           for _ in range(length)]
        else:
            self.genome = genome
        self.fitness = fitness.default_fitness(FITNESS_FUNCTION.maximise)
        self.phenotype = None
        self.used_codons = 0
        self.compiled_phenotype = None

    def __lt__(self, other):
        if FITNESS_FUNCTION.maximise:
            return self.fitness < other.fitness
        else:
            return other.fitness < self.fitness

    def __str__(self):
        train_fit = str(self.fitness)
        if hasattr(FITNESS_FUNCTION, "test"):
            test_fit = str(FITNESS_FUNCTION.test(self.phenotype))
        else:
            test_fit = train_fit
        return ("Individual: " + str(self.phenotype) + "; " +
                train_fit + "; " + test_fit)

    def generate(self, grammar):
        self.phenotype, self.used_codons = grammar.generate(self.genome)

    def evaluate(self, fitness):
        """Evaluates phenotype in fitness function and sets fitness"""
        self.fitness = fitness(self.phenotype)

class DTIndividual(Individual):
    """An individual with a derivation tree genome instead of
    integer-array."""
    def __init__(self, genome, grammar):
        if genome == None:
            self.genome = dt.random_dt(grammar)
        else:
            self.genome = genome
        self.fitness = fitness.default_fitness(FITNESS_FUNCTION.maximise)
        self.phenotype = None
        self.compiled_phenotype = None
        self.used_codons = 0

    def generate(self, grammar):
        # This is a bit of a hack: the number of nodes in the dt
        # doesn't really correspond to the number of codons that would
        # be used. But it's useful when printing stats to use this
        # variable.
        self.used_codons = len(list(dt.traverse(self.genome)))
        self.phenotype = dt.derived_str(self.genome, grammar)

def initialise_population(size, grammar=None):
    """Create a population of Individuals of the given size. If
    grammar is passed-in, create DTIndividuals instead."""
    if grammar:
        return [DTIndividual(None, grammar) for _ in range(size)]
    else:
        return [Individual(None) for _ in range(size)]

def print_header():
    print("# generation evaluations best_fitness best_used_codons " +
          "mean_fitness stddev_fitness mean_used_codons stddev_used_codons " +
          "number_invalids mean_genome_length best_phenotype")

def print_stats(generation, individuals):
    """Print the statistics for the generation and individuals"""
    def ave(values):
        """Return the average of the values """
        return float(sum(values))/len(values)
    def std(values, ave):
        """Return the standard deviation of the values and average """
        return math.sqrt(float(sum((value-ave)**2 for value in values))/len(values))

    valid_inds = [i for i in individuals if i.phenotype is not None]
    ninvalids = len(individuals) - len(valid_inds)
    if len(valid_inds) == 0:
        fitness_vals = [0]
        used_codon_vals = [0]
    else:
        fitness_vals = [i.fitness for i in valid_inds]
        used_codon_vals = [i.used_codons for i in valid_inds]
    len_vals = [len(i.genome) for i in individuals]
    ave_fit = ave(fitness_vals)
    std_fit = std(fitness_vals, ave_fit)
    ave_used_codons = ave(used_codon_vals)
    std_used_codons = std(used_codon_vals, ave_used_codons)
    ave_len = ave(len_vals)
    std_len = std(len_vals, ave_len)
    print("{0} {1} {2} {3} {4:.2f} {5:.2f} {6:.2f} {7:.2f} {8:.2f} {9:.2f} {10} : {11}"
          .format(generation, GENERATION_SIZE * generation,
                  individuals[0].fitness, individuals[0].used_codons,
                  ave_fit, std_fit, ave_used_codons, std_used_codons,
                  ave_len, std_len, ninvalids,
                  individuals[0].phenotype))

def int_flip_mutation(individual):
    """Mutate the individual by randomly chosing a new int with
    probability p_mut. Works per-codon, hence no need for
    "within_used" option."""
    # in case the input individual is later re-used as a parent:
    # we must not modify it here.
    individual = copy.deepcopy(individual)
    for i in range(len(individual.genome)):
        if random.random() < MUTATION_PROBABILITY:
            individual.genome[i] = random.randint(0, CODON_SIZE)
    return individual

def dt_mutation(x, grammar):
    """Given an individual whose genome is a DT, return a new
    individual by mutation."""
    if random.random() < MUTATION_PROBABILITY:
        return DTIndividual(dt.dt_mutation(copy.deepcopy(x.genome), grammar), grammar)
    else:
        return x

# Two selection methods: tournament and truncation
def tournament_selection(population, tournament_size=3):
    """Given an entire population, draw <tournament_size> competitors
    randomly and return the best."""
    winners = []
    while len(winners) < GENERATION_SIZE:
        competitors = random.sample(population, tournament_size)
        competitors.sort(reverse=True)
        winners.append(competitors[0])
    return winners

def truncation_selection(population, proportion=0.5):
    """Given an entire population, return the best <proportion> of
    them."""
    population.sort(reverse=True)
    cutoff = int(len(population) * float(proportion))
    return population[0:cutoff]

def onepoint_crossover(p_0, p_1, within_used=True):
    """Given two individuals, create two children using one-point
    crossover and return them."""
    # Get the chromosomes
    c_p_0, c_p_1 = p_0.genome, p_1.genome
    # Uniformly generate crossover points. If within_used==True,
    # points will be within the used section.
    len_p_0, len_p_1 = len(c_p_0), len(c_p_1)
    if within_used:
        # -1 to get last index in array; min() in case of wraps: used > len
        max_p_0, max_p_1 = min(p_0.used_codons, len_p_0 - 1), min(p_1.used_codons, len_p_1 - 1)
    else:
        max_p_0, max_p_1 = len_p_0 - 1, len_p_1 - 1
    pt_p_0, pt_p_1 = random.randint(1, max_p_0), random.randint(1, max_p_1)
    # Make new chromosomes by crossover: these slices perform copies
    if random.random() < CROSSOVER_PROBABILITY:
        c_0 = c_p_0[:pt_p_0] + c_p_1[pt_p_1:]
        c_1 = c_p_1[:pt_p_1] + c_p_0[pt_p_0:]
    else:
        c_0, c_1 = c_p_0[:], c_p_1[:]
    # Put the new chromosomes into new individuals
    return [Individual(c_0), Individual(c_1)]

def dt_crossover(t, s, grammar):
    """Given individuals whose genomes are DTs, return new individuals
    formed by crossover."""
    newtrees = copy.deepcopy(t.genome), copy.deepcopy(s.genome)
    dt.dt_crossover(newtrees[0], newtrees[1], grammar)
    return [DTIndividual(newtree, grammar) for newtree in newtrees]

def evaluate_fitness(individuals, grammar, fitness_function):
    """Perform the mapping for each individual """
    for ind in individuals:
        ind.generate(grammar)
        if ind.phenotype != None:
            if not hasattr(fitness_function, "COEVOLUTION") or \
                    not fitness_function.COEVOLUTION:
                ind.evaluate(fitness_function)
    if hasattr(fitness_function, "COEVOLUTION") and fitness_function.COEVOLUTION:
        fitness_function.__call__(individuals)

def interactive_evaluate_fitness(individuals, grammar, callback):
    """Used for interactive evolution. Perform mapping and set dummy fitness"""
    evaluate_fitness(individuals, grammar, lambda x: 0.0)
    fitness_values = callback()
    for i, individual in enumerate(individuals):
        if individual.phenotype != None:
            individual.fitness = fitness_values[i]

def generational_replacement(new_pop, individuals):
    """Return new pop. The ELITE_SIZE best individuals are appended
    to new pop if they are better than the worst individuals in new
    pop"""
    individuals.sort(reverse=True)
    for ind in individuals[:ELITE_SIZE]:
        new_pop.append(copy.copy(ind))
    new_pop.sort(reverse=True)
    return new_pop[:GENERATION_SIZE]

def steady_state_replacement(new_pop, individuals):
    """Return individuals. If the best of new pop is better than the
    worst of individuals it is inserted into individuals"""
    individuals.sort(reverse=True)
    individuals[-1] = max(new_pop + individuals[-1:])
    return individuals

def step(individuals, grammar, crossover, mutation,
         replacement, selection, fitness_function, best_ever):
    """Return individuals and best ever individual from a step of
    the EA iteration"""
    #Select parents
    parents = selection(individuals)
    #Crossover parents and add to the new population
    new_pop = []
    while len(new_pop) < GENERATION_SIZE:
        new_pop.extend(crossover(*random.sample(parents, 2)))
    #Mutate the new population
    new_pop = list(map(mutation, new_pop))
    #Evaluate the fitness of the new population
    evaluate_fitness(new_pop, grammar, fitness_function)
    #Replace the sorted individuals with the new populations
    individuals = replacement(new_pop, individuals)
    best_ever = max(best_ever, max(individuals))
    return individuals, best_ever

def search_loop(max_generations, individuals, grammar,
                crossover, mutation,
                replacement, selection, fitness_function):
    """Loop over max generations"""
    #Evaluate initial population
    evaluate_fitness(individuals, grammar, fitness_function)
    best_ever = max(individuals)
    individuals.sort(reverse=True)
    print_stats(1, individuals)
    for generation in range(2, (max_generations+1)):
        individuals, best_ever = step(
            individuals, grammar, crossover, mutation,
            replacement, selection, fitness_function, best_ever)
        print_stats(generation, individuals)
    return best_ever

VERBOSE = False
DERIVATION_TREE_GENOME = False
CODON_SIZE = 127
ELITE_SIZE = 1
POPULATION_SIZE = 200
GENERATION_SIZE = 200
GENERATIONS = 40
MUTATION_PROBABILITY = 0.01
CROSSOVER_PROBABILITY = 0.7
#GRAMMAR_FILE, FITNESS_FUNCTION = "grammars/boolean_hof.bnf", fitness.BooleanProblem(5, lambda x: ~(x[0] ^ x[1] ^ x[2] ^ x[3] ^ x[4]))
#GRAMMAR_FILE, FITNESS_FUNCTION = "grammars/boolean.bnf", fitness.BooleanProblem(5, lambda x: ~(x[0] ^ x[1] ^ x[2] ^ x[3] ^ x[4]))
#GRAMMAR_FILE, FITNESS_FUNCTION = "grammars/letter.bnf", StringMatch("golden")
#GRAMMAR_FILE, FITNESS_FUNCTION = "grammars/arithmetic.pybnf", fitness.MaxFitness()
GRAMMAR_FILE, FITNESS_FUNCTION = "grammars/symbolic_regression_2d.bnf", fitness.SymbolicRegressionFitnessFunction("data/fagan_train.dat", "data/fagan_test.dat")
#GRAMMAR_FILE, FITNESS_FUNCTION = "grammars/boolean_hof.bnf", fitness.BooleanProblemGeneral([2, 3], [5], lambda x: reduce((lambda u, v: u ^ v), x))

def mane():
    """Run program"""
    # Read grammar
    bnf_grammar = Grammar(GRAMMAR_FILE, nvars=len(FITNESS_FUNCTION.train_X))
    if VERBOSE:
        print(bnf_grammar)
    print_header()
    # Genetic operators: initialise, crossover, mutation
    if DERIVATION_TREE_GENOME:
        crossover = lambda x, y: dt_crossover(x, y, bnf_grammar)
        mutation = lambda x: dt_mutation(x, bnf_grammar)
        individuals = initialise_population(POPULATION_SIZE, bnf_grammar)
    else:
        crossover, mutation = onepoint_crossover, int_flip_mutation
        individuals = initialise_population(POPULATION_SIZE)
    # Loop
    best_ever = search_loop(GENERATIONS, individuals, bnf_grammar,
                            crossover, mutation,
                            generational_replacement, tournament_selection,
                            FITNESS_FUNCTION)
    print("Best " + str(best_ever))
    return best_ever

if __name__ == "__main__":
    import getopt
    try:
        # FIXME help option
        OPTS, ARGS = getopt.getopt(sys.argv[1:], "vdp:g:e:m:x:b:f:",
                                   ["verbose", "derivation_tree",
                                    "population", "generations",
                                    "elite_size", "mutation", "crossover",
                                    "bnf_grammar", "fitness_function"])
    except getopt.GetoptError as err:
        print(str(err))
        # FIXME usage
        sys.exit(2)
    for opt, arg in OPTS:
        if opt in ("-v", "--verbose"):
            VERBOSE = True
        elif opt in ("-d", "--derivation_tree"):
            DERIVATION_TREE_GENOME = True
        elif opt in ("-p", "--population"):
            POPULATION_SIZE = int(arg)
            GENERATION_SIZE = int(arg)
        elif opt in ("-g", "--generations"):
            GENERATIONS = int(arg)
        elif opt in ("-e", "--elite_size"):
            ELITE_SIZE = int(arg)
        elif opt in ("-m", "--mutation"):
            MUTATION_PROBABILITY = float(arg)
        elif opt in ("-x", "--crossover"):
            CROSSOVER_PROBABILITY = float(arg)
        elif opt in ("-b", "--bnf_grammar"):
            GRAMMAR_FILE = arg
        elif opt in ("-f", "--fitness_function"):
            FITNESS_FUNCTION = eval(arg)
        else:
            assert False, "unhandled option"
    mane()
