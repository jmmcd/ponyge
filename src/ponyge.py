#! /usr/bin/env python

# PonyGE
# Copyright (c) 2009 Erik Hemberg and James McDermott
# Hereby licensed under the GNU GPL v3.
""" Small GE implementation """

import sys, copy, re, random, math, operator

class Grammar(object):
    """ Context Free Grammar """
    NT = "NT" # Non Terminal
    T = "T" # Terminal

    def __init__(self, file_name):
        if file_name.endswith("pybnf"):
            self.python_mode = True
        else:
            self.python_mode = False
        self.rules = {}
        self.non_terminals, self.terminals = set(), set()
        self.start_rule = None

        self.read_bnf_file(file_name)

    def read_bnf_file(self, file_name):
        """Read a grammar file in BNF format"""
        # <.+?> Non greedy match of anything between brackets
        non_terminal_pattern = "(<.+?>)"
        rule_separator = "::="
        production_separator = "|"

        # Read the grammar file
        for line in open(file_name, 'r'):
            if not line.startswith("#") and line.strip() != "":
                # Split rules. Everything must be on one line
                if line.find(rule_separator):
                    lhs, productions = line.split(rule_separator)
                    lhs = lhs.strip()
                    if not re.search(non_terminal_pattern, lhs):
                        raise ValueError("lhs is not a NT:", lhs)
                    self.non_terminals.add(lhs)
                    if self.start_rule == None:
                        self.start_rule = (lhs, self.NT)
                    # Find terminals
                    tmp_productions = []
                    for production in [production.strip()
                                       for production in
                                       productions.split(production_separator)]:
                        tmp_production = []
                        if not re.search(non_terminal_pattern, production):
                            self.terminals.add(production)
                            tmp_production.append((production, self.T))
                        else:
                            # Match non terminal or terminal pattern
                            # TODO does this handle quoted NT symbols?
                            for value in re.findall("<.+?>|[^<>]*", production):
                                if value != '':
                                    if not re.search(non_terminal_pattern,
                                                     value):
                                        symbol = (value, self.T)
                                        self.terminals.add(value)
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

    def __str__(self):
        return "%s %s %s %s" % (self.terminals, self.non_terminals,
                                self.rules, self.start_rule)

    def generate(self, _input, max_wraps=2):
        """Map input via rules to output. Returns output and used_input"""
        used_input = 0
        wraps = 0
        output = []
        production_choices = []

        unexpanded_symbols = [self.start_rule]
        while (wraps < max_wraps) and (len(unexpanded_symbols) > 0):
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
    """ Create correct python syntax.

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

def eval_or_exec(expr):
    """ Use eval or exec to interpret expr.

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
        retval = default_fitness(FITNESS_FUNCTION.maximise)
    return retval

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
    """Maximisation with python evaluation."""
    maximise = True
    def __call__(self, candidate):
        return eval_or_exec(candidate)

class Individual(object):
    """A GE individual"""
    def __init__(self, genome, length=100):
        if genome == None:
            self.genome = [random.randint(0, CODON_SIZE)
                           for _ in range(length)]
        else:
            self.genome = genome
        self.fitness = default_fitness(FITNESS_FUNCTION.maximise)
        self.phenotype = None
        self.used_codons = 0
        self.compiled_phenotype = None

    def __lt__(self, other):
        if FITNESS_FUNCTION.maximise:
            return self.fitness < other.fitness
        else:
            return other.fitness < self.fitness

    def __str__(self):
        return ("Individual: " +
                str(self.phenotype) + "; " + str(self.fitness))

    def evaluate(self, fitness):
        """ Evaluates phenotype in fitness function and sets fitness"""
        self.fitness = fitness(self.phenotype)

def initialise_population(size=10):
    """Create a popultaion of size and return"""
    return [Individual(None) for _ in range(size)]

def print_stats(generation, individuals):
    """Print the statistics for the generation and individuals"""
    def ave(values):
        """ Return the average of the values """
        return float(sum(values))/len(values)
    def std(values, ave):
        """ Return the standard deviation of the values and average """
        return math.sqrt(float(sum((value-ave)**2 for value in values))/len(values))

    valid_inds = [i for i in individuals if i.phenotype is not None]
    if len(valid_inds) == 0:
        fitness_vals = [0]
        used_codon_vals = [0]
    else:
        fitness_vals = [i.fitness for i in valid_inds]
        used_codon_vals = [i.used_codons for i in valid_inds]
    ave_fit = ave(fitness_vals)
    std_fit = std(fitness_vals, ave_fit)
    ave_used_codons = ave(used_codon_vals)
    std_used_codons = std(used_codon_vals, ave_used_codons)
    print("Gen:%d evals:%d ave:%.2f+-%.3f aveUsedC:%.2f+-%.3f %s" % (
            generation, (GENERATION_SIZE*generation), ave_fit, std_fit,
            ave_used_codons, std_used_codons, individuals[0]))

def default_fitness(maximise):
    """ Return default fitness given maximization of minimization"""
    if maximise:
        return -100000.0
    else:
        return 100000.0

def int_flip_mutation(individual):
    """Mutate the individual by randomly chosing a new int with
    probability p_mut. Works per-codon, hence no need for
    "within_used" option."""
    for i in range(len(individual.genome)):
        if random.random() < MUTATION_PROBABILITY:
            individual.genome[i] = random.randint(0, CODON_SIZE)
    return individual

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
    if within_used:
        max_p_0, max_p_1 = p_0.used_codons, p_1.used_codons
    else:
        max_p_0, max_p_1 = len(c_p_0), len(c_p_1)
    pt_p_0, pt_p_1 = random.randint(1, max_p_0), random.randint(1, max_p_1)
    # Make new chromosomes by crossover: these slices perform copies
    if random.random() < CROSSOVER_PROBABILITY:
        c_0 = c_p_0[:pt_p_0] + c_p_1[pt_p_1:]
        c_1 = c_p_1[:pt_p_1] + c_p_0[pt_p_0:]
    else:
        c_0, c_1 = c_p_0[:], c_p_1[:]
    # Put the new chromosomes into new individuals
    return [Individual(c_0), Individual(c_1)]

def evaluate_fitness(individuals, grammar, fitness_function):
    """ Perform the mapping for each individual """
    for ind in individuals:
        ind.phenotype, ind.used_codons = grammar.generate(ind.genome)
        if ind.phenotype != None:
            if not hasattr(fitness_function, "COEVOLUTION") or \
                    not fitness_function.COEVOLUTION:
                ind.evaluate(fitness_function)
    if hasattr(fitness_function, "COEVOLUTION") and fitness_function.COEVOLUTION:
        fitness_function.__call__(individuals)

def interactive_evaluate_fitness(individuals, grammar, callback):
    """ Used for interactive evolution. Perform mapping and set dummy fitness"""
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

def step(individuals, grammar, replacement, selection, fitness_function, best_ever):
    """Return individuals and best ever individual from a step of
    the EA iteration"""
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
    best_ever = max(best_ever, max(individuals))
    return individuals, best_ever

def search_loop(max_generations, individuals, grammar, replacement, selection, fitness_function):
    """Loop over max generations"""
    #Evaluate initial population
    evaluate_fitness(individuals, grammar, fitness_function)
    best_ever = max(individuals)
    individuals.sort(reverse=True)
    print_stats(1, individuals)
    for generation in range(2, (max_generations+1)):
        individuals, best_ever = step(
            individuals, grammar, replacement, selection, fitness_function, best_ever)
        print_stats(generation, individuals)
    return best_ever

VERBOSE = False
CODON_SIZE = 127
ELITE_SIZE = 1
POPULATION_SIZE = 100
GENERATION_SIZE = 100
GENERATIONS = 30
MUTATION_PROBABILITY = 0.01
CROSSOVER_PROBABILITY = 0.7
#GRAMMAR_FILE, FITNESS_FUNCTION = "grammars/hofBoolean.pybnf", EvenNParityFitness(3)
GRAMMAR_FILE, FITNESS_FUNCTION = "grammars/letter.bnf", StringMatch("golden")
#GRAMMAR_FILE, FITNESS_FUNCTION = "grammars/arithmetic.pybnf", MaxFitness()
#GRAMMAR_FILE, FITNESS_FUNCTION = "grammars/boolean.pybnf", XORFitness()

def mane():
    """ Run program """
    # Read grammar
    bnf_grammar = Grammar(GRAMMAR_FILE)
    if VERBOSE:
        print(bnf_grammar)
    # Create Individuals
    individuals = initialise_population(POPULATION_SIZE)
    # Loop
    best_ever = search_loop(GENERATIONS, individuals, bnf_grammar,
                            generational_replacement, tournament_selection,
                            FITNESS_FUNCTION)
    print("Best " + str(best_ever))

if __name__ == "__main__":
    import getopt
    try:
        #FIXME help option
        print(sys.argv)
        OPTS, ARGS = getopt.getopt(sys.argv[1:], "vp:g:e:m:x:b:f:",
                                   ["verbose", "population", "generations",
                                    "elite_size", "mutation", "crossover",
                                    "bnf_grammar", "fitness_function"])
    except getopt.GetoptError as err:
        print(str(err))
        #FIXME usage
        sys.exit(2)
    for opt, arg in OPTS:
        if opt in ("-v", "--verbose"):
            VERBOSE = True
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
            assert False, "unhandeled option"
    mane()
