#! /usr/bin/env python

# PonyGE
# Copyright (c) 2009 Erik Hemberg and James McDermott
# Hereby licensed under the GNU GPL v3.


import sys
import re
import random

class Grammar(object):
    # Non Terminal 
    NT = "NT"
    # Terminal 
    T = "T"

    def __init__(self, file_name):
        if file_name.endswith("pybnf"):
            self.python_mode = True
        else:
            self.python_mode = False
        self.readBNFFile(file_name)

    # Read Grammar file
    def readBNFFile(self, file_name):
        """Read a grammar file in BNF format and return the rules"""
        NON_TERMINAL_PATTERN = "(<.+?>)"
        RULE_SEPARATOR = "::="
        PRODUCTION_SEPARATOR = "|"

        infile = open(file_name, 'r')
        self.rules = {}
        # Non-Terminal set
        self.non_terminals = set()
        # Terminal set
        self.terminals = set()
        self.start_rule = None
        # Read the grammar file
        for line in infile:
            # Not read comment lines
            if not line.startswith("#"):
                # Split rules. Everything must be on one line
                #TODO Avoid everything on one line
                #TODO find and match instead of explicit split
                if line.find(RULE_SEPARATOR):
                    lhs, productions = line.split(RULE_SEPARATOR)
                    lhs = lhs.strip()
                    if self.start_rule == None:
                        #TODO Need to make sure it is NT
                        self.start_rule = (lhs, self.NT)
                    self.non_terminals.add(lhs)
                    # Split productions
                    productions = productions.split(PRODUCTION_SEPARATOR)
                    productions = [production.strip() for production in productions]
                    # Find terminals
                    tmp_productions = []
                    for production in productions:
                        # <.+?> Non greedy match of anything between brackets
                        found = re.search(NON_TERMINAL_PATTERN, production)
                        if not found:
                            self.terminals.add(production)
                            symbol = (production, self.T)
                            tmp_production = symbol
                        else:
                            # Match non terminal or terminal pattern
                            # TODO does this handle quoted NT symbols
                            pattern = "<.+?>|[^<>]*";
                            # Find Non-Terminals
                            found = re.findall(pattern, production)
                            tmp_production = []
                            for f in found:
                                if f != '':
                                    nt = re.search(NON_TERMINAL_PATTERN, f)
                                    if not nt:
                                        symbol = (f, self.T)
                                    else:
                                        symbol = (f, self.NT)
                                    tmp_production.append(symbol)
                        tmp_productions.append(tmp_production)
                    # Create a rule
                    if not self.rules.has_key(lhs):
                        self.rules[lhs] = tmp_productions
                    else:
                        print "WARNING:",__name__, "readBNFFile",  lhs, "should be unique"

        
    # Map individual
    def generate(self, input, max_wraps=2):
        """Map input via rules to output"""
        cnt = 0
        wraps = 0
        output = []
        # Stack of symbols to expand
        unexpanded_symbols = []
        unexpanded_symbols.append(self.start_rule)
        # Should I write it recursivly??!! (Good to test speed difference)
        while (cnt < len(input)) and (wraps < max_wraps) and (len(unexpanded_symbols) > 0):
            # Wrap
            if cnt == len(input):
                wraps += 1
                cnt = 0
            # Get a prodcution
            current_symbol = unexpanded_symbols.pop(0)
            # Get output if it is a terminal        
            if current_symbol[1] != self.NT:
                output.append(current_symbol[0])
            else:
                # Get production choices
                production_choices = self.rules[current_symbol[0]]
                # Select a production
                current_production = input[cnt] % len(production_choices)
                # Use input if there was more then 1 choice
                if len(production_choices) > 1:
                    cnt += 1
                # Read left to right, add the current productions to stack
                tmp_list = []
                tmp_choice = production_choices[current_production]
                # Stupid python treats a lonely tuple as a list in a for-loop
                # so it loops over the elements in the tuple
                # TODO make sure all the productions are lists
                if len(tmp_choice) == 2 and (tmp_choice[1] == self.T or tmp_choice[1] == self.NT):
                    tmp_tuple = (tmp_choice[0], tmp_choice[1])
                    tmp_list.append(tmp_tuple)
                else:
                    for e in tmp_choice:
                        tmp_list.append(e)
                for e in unexpanded_symbols:
                    tmp_list.append(e)
                unexpanded_symbols = tmp_list

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
    #Initial fitness, penalise too long strings
    fitness = max(len(target), len(output))
    for (t,o) in zip(target, output):
        # If target is matching output decrease fitness
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
    """A GE 8 bit individual"""
    def __init__(self, genome, length=100):
        if genome == None:
            self.genome = [random.randint(0, CODON_SIZE) 
                           for i in range(length)]
        else:
            self.genome = genome
        self.fitness = -1
        self.phenotype = None

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
    input = individual.genome
    # Mutate the input
    for i in range(len(input)):
        # Check mutation probability
        if random.random() < p_mut:
            input[i] = random.randint(0,CODON_SIZE)
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
    pt_p, pt_q = random.randint(0, len(pc)), random.randint(0, len(qc))
    # Make new chromosomes by crossover: these slices perform copies
    c = pc[:pt_p] + qc[pt_q:]
    d = qc[:pt_q] + pc[pt_p:]
    # Put the new chromosomes into new individuals
    return [Individual(c), Individual(d)]

# Loop 
def search_loop(max_generations, individuals, grammar):
    """Loop over max generations"""
    for generation in range(max_generations):
        print("Gen:", generation)
        # Perform the mapping for each individual
        for i in range(len(individuals)):
            ind = individuals[i]
            ind.phenotype = grammar.generate(ind.genome)
            if ind.phenotype != None:
                # ind.evaluate(lambda x: string_match("geva", x))
                ind.evaluate(xor_fitness)

        #TODO sort individuals before printing
        print_individuals(individuals)
            
        # Perform selection, crossover, and mutation
        parents = truncation_selection(individuals, 0.5)
        new_pop = []
        #TODO write generational and steady state, and elites
        while len(new_pop) < len(individuals):
            two_parents = random.sample(parents, 2)
            new_pop.extend(onepoint_crossover(*two_parents))
        for i in range(len(new_pop)):
            new_pop[i] = int_flip_mutation(new_pop[i], 0.05)
        individuals = new_pop

#Codon size used for the individuals
CODON_SIZE = 127 
# Run program
def main():
    # Read grammar
    bnf_grammar = Grammar("grammars/boolean.pybnf")
    # Create Individuals
    individuals = initialise_population(10)
    # Loop
    search_loop(10, individuals, bnf_grammar)

if __name__ == "__main__":
    main()
