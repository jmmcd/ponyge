#! /usr/bin/env python

# PonyGE
# Copyright (c) 2009 Erik Hemberg and James McDermott
# Hereby licensed under the GNU GPL v3.


import sys
import re
import random

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
            if not line.startswith("#"):
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
                    if not self.rules.has_key(lhs):
                        self.rules[lhs] = tmp_productions
                    else:
                        raise ValueError("lhs should be unique", lhs)
                else:
                    raise ValueError("Each rule must be on one line")
        
    def generate(self, input, max_wraps=2):
        """Map input via rules to output"""
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
                # Get production choices
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
            return None

        output = "".join(output)
        #Create correct python syntax
        if self.python_mode:
            counter = 0
            for char in output:
                if char == "{":
                    counter += 1
                elif char == "}":
                    counter -= 1
                tabstr = "\n" + "  " * counter
                if char == "{" or char == "}":
                    output = output.replace(char, tabstr, 1)
            output = "\n".join([line for line in output.split("\n") 
                                if line.strip() != ""])

        return output

# String-match fitness function
def string_match(target, output):
    """Fitness function for matching a string.  Takes an output string
    and return fitness. Penalises output that is not the same length
    as the target"""
    fitness = max(len(target), len(output))
    #Loops as long as the min(target, output) 
    for (t,o) in zip(target, output):
        if t == o:
            fitness -= 1
    return fitness

# XOR fitness function with python evaluation
def xor_fitness(candidate):
    def xor(x, y):
        return (x and not y) or (y and not x)
    f = eval(candidate)
    fitness = 0
    for x in [False, True]:
        for y in [False, True]:
            if f(x, y) != xor(x, y):
                fitness += 1
    return fitness

class Individual(object):
    """A GE individual"""
    def __init__(self, genome, fitness=None, phenotype=None, length=100):
        if genome == None:
            self.genome = [random.randint(0, CODON_SIZE) 
                           for i in range(length)]
        else:
            self.genome = genome
        self.fitness = fitness
        self.phenotype = phenotype
    
    def __cmp__(self, other):
        #-1*cmp for maximization
        #TODO variable for minimization or maximization
        #TODO None seems to be lowest in minimization
        return cmp(self.fitness, other.fitness)

    def __str__(self):
        return ("Individual: " + 
                str(self.phenotype) + "; " + str(self.fitness))

    def evaluate(self, fitness):
        self.fitness = fitness(self.phenotype)

# Initialize population
def initialise_population(size=10):
    """Create a popultaion of size and return"""
    individuals = []
    for cnt in range(size):
        individuals.append(Individual(None))

    return individuals

# Write data
def print_individuals(individuals):
    """Print the data of the individuals"""
    for individual in individuals:
        print(individual)

# Int flip mutation
def int_flip_mutation(individual, p_mut):
    """Mutate the individual by randomly chosing a new int with 
    probability p_mut"""
    for i in range(len(individual.genome)):
        if random.random() < p_mut:
            individual.genome[i] = random.randint(0,CODON_SIZE)
    return individual

# Two selection methods: tournament and truncation
def tournament_selection(population, tournament_size=3):
    """Given an entire population, draw <tournament_size> competitors
    randomly and return the best."""
    competitors = random.sample(population, tournament_size)
    return competitors.sort()[0]

def truncation_selection(population, proportion):
    """ Given an entire population, return the best <proportion> of
    them."""
    population.sort(reverse=True)
    cutoff = int(len(population) * float(proportion))
    return population[0:cutoff]

# Crossover
def onepoint_crossover(p, q):
    """Given two individuals, create two children using one-point
    crossover and return them."""    
    # Get the chromosomes
    pc, qc = p.genome, q.genome
    # Uniformly generate crossover points
    pt_p, pt_q = random.randint(1, len(pc)), random.randint(1, len(qc))
    # Make new chromosomes by crossover: these slices perform copies
    c = pc[:pt_p] + qc[pt_q:]
    d = qc[:pt_q] + pc[pt_p:]
    # Put the new chromosomes into new individuals
    return [Individual(c), Individual(d)]

def evaluate_fitness(individuals, grammar):
    # Perform the mapping for each individual
    for individual in individuals:
        individual.phenotype = grammar.generate(individual.genome)
        if individual.phenotype != None:
            individual.evaluate(lambda x: string_match("geva", x))
            #individual.evaluate(xor_fitness)

def generational_replacement(new_pop, individuals):
    #TODO make pythonic map()? not loop
    for ind in individuals[:ELITE_SIZE]:
        new_pop.append(Individual(ind.genome, ind.fitness, ind.phenotype))
    new_pop.sort()
    return new_pop[:GENERATION_SIZE]

def steady_state_replacement(new_pop, individuals):
    individuals[-1] = max(new_pop + individuals[-1:])
    return individuals

# Loop 
def search_loop(max_generations, individuals, grammar):
    """Loop over max generations"""
    #Evaluate initial population
    #TODO look like pseudo code
    #TODO handle initialisation nicely
    print("Gen:", -1)
    evaluate_fitness(individuals, grammar)
    individuals.sort()
    #print_individuals(individuals)
    # Perform selection, crossover, mutation, evaluation and replacement
    for generation in range(max_generations):
        print("Gen:", generation)

        parents = truncation_selection(individuals, 0.5)

        new_pop = []
        while len(new_pop) < GENERATION_SIZE:
            two_parents = random.sample(parents, 2)
            if random.random() < CROSSOVER_PROBABILITY:
                two_parents = onepoint_crossover(*two_parents)
            else:
                #TODO make pythonic
                for i in range(len(two_parents)):
                    two_parents[i] = Individual(two_parents[i].genome)
            new_pop.extend(two_parents)

        for i in range(len(new_pop)):
            new_pop[i] = int_flip_mutation(new_pop[i], MUTATION_PROBABILITY)
        
        evaluate_fitness(new_pop, grammar)

        individuals.sort()
        individuals = generational_replacement(new_pop, individuals)
#        individuals = steady_state_replacement(new_pop, individuals)

#        print_individuals(individuals)
        print individuals[0]
        

#Codon size used for the individuals
#TODO can the functions be structured in a more sensible manner to
#make the file more readable
#TODO initial size parameter
CODON_SIZE = 127 
ELITE_SIZE = 1
POPULATION_SIZE = 100
GENERATION_SIZE = 100
#GENERATION_SIZE = 2
#TODO should we count fitness evaluations
GENERATIONS = 10
#GRAMMAR_FILE = "grammars/boolean.pybnf"
GRAMMAR_FILE = "grammars/letter.bnf"
MUTATION_PROBABILITY = 0.05
CROSSOVER_PROBABILITY = 0.9
# Run program
def main():
    # Read grammar
    bnf_grammar = Grammar(GRAMMAR_FILE)
    # Create Individuals
    individuals = initialise_population(POPULATION_SIZE)
    # Loop
    #TODO Look like functioncall, with all paraeters in one?
    search_loop(GENERATIONS, individuals, bnf_grammar)

if __name__ == "__main__":
    main()
