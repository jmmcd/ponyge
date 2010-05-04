Grammatical evolution (GE) is an evolutionary algorithm which uses
formal grammars, written in BNF, to define the search space. PonyGE is
a small (one source file!) but functional implementation of GE in
Python. It's intended as an advertisement and a starting-point for
those new to GE, a reference for implementors and researchers, a
rapid-prototyping medium for our own experiments, and a Python
workout. And a pony.

PonyGE is copyright (C) 2009-2010 Erik Hemberg
<erik.hemberg@gmail.com> and James McDermott
<jamesmichaelmcdermott@gmail.com>.


Requirements
------------

PonyGE runs under Python 2.4+ and Python 3.x. The most interesting
example problem (see L-System below) requires Python 2.6+ or 3.x and
Tkinter.


Running PonyGE
--------------

We don't provide any setup script. You can run an example problem (the
default is String-match, see below) just by saying:

$ ./ponyge.py

Each line of the output corresponds to a generation in the evolution,
and tells you the generation number, number of fitness evaluations
which have taken place, average fitness with standard deviation,
average number of codons used (see any GE paper, eg those referenced
below, for definition) with standard deviation, and the best
individual found so far. At the end, the best individual is printed
out again.


Example Problems
----------------


String-match
------------

The grammar specifies words as lists of vowels and consonants. The aim
is to match a target word. This is the default problem: as you can see
in ponyge.py, the necessary grammar and fitness function are specified
by default:

GRAMMAR_FILE = "grammars/letter.bnf"
FITNESS_FUNCTION = StringMatch("golden")


Max
---

The grammar specifies legal ways of combining numbers using arithmetic
operations. The aim is to create the largest value possible. Use the
following grammar and fitness function (in ponyge.py):

GRAMMAR_FILE = "grammars/arithmetic.pybnf"
FITNESS_FUNCTION = MaxFitness()


XOR
---

A small version of the standard genetic programming Even-N parity
benchmark. The grammar specifies two inputs, x and y, and allows them
to be combined using AND, OR, and NOT. The aim is to evolve the XOR
function. Use the following grammar and fitness function:

GRAMMAR_FILE = "grammars/boolean.pybnf"
FITNESS_FUNCTION = XORFitness()


L-System
--------

The most interesting example. Run it like this (no need to make any
source code changes):

$ ./gui.py

You'll be presented with a GUI showing nine drawings in a 3x3 grid of
cells. You click on the ones you like, and click "Next" (or hit space)
to iterate the algorithm. The drawings are made using a custom
L-system whose possible forms are specified by the grammar. The files
gui.py, drawing.py, and lsystem.py all belong to this example.


Reference
---------

Michael O'Neill and Conor Ryan, "Grammatical Evolution: Evolutionary
Automatic Programming in an Arbitrary Language", Kluwer Academic
Publishers, 2003.

Michael O'Neill, Erik Hemberg, Conor Gilligan, Elliott Bartley, and
James McDermott, "GEVA: Grammatical Evolution in Java", ACM
SIGEVOlution, 2008. http://portal.acm.org/citation.cfm?id=1527066. Get
GEVA: http://ncra.ucd.ie/Site/GEVA.html