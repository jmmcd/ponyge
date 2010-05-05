#!/usr/local/bin/python
#
# $Id: //projects/lsystem/lsystem.py#1 $ $Date: 2002/07/29 $

"""
Lindenmeyer system (L-system) simulator in Python.
"""

__program__ = 'lsystem'
__version__ = '1.0'
__author__ = 'Erik Max Francis <max@alcyone.com>'
__copyright__ = 'Copyright (C) 2002 Erik Max Francis'
__license__ = 'GPL'


class LSystem:

    """The standard L-system."""
    
    axiom = None
    rules = None

    def __init__(self, axiom=None, rules=None):
        if axiom is not None:
            self.axiom = axiom
        if rules is not None:
            self.rules = rules
        self.string = self.axiom
        self.generation = 0
        self.done = 0

    def reset(self):
        """Reset the system to its axiom."""
        self.string = self.axiom

    def step(self):
        """Execute one generation of the system."""
        self.generation += 1
        newString = ''
        stringLen = len(self.string)
        i = 0
        while i < stringLen:
            for input, output in self.rules:
                if self.string[i:i + len(input)] == input:
                    newString += output
                    i += len(input)
                    break
            else:
                newString += self.string[i]
                i += 1
        if self.string == newString:
            self.done = 1
        self.string = newString

    def __len__(self):
        return len(self.string)

    def __getitem__(self, index):
        return self.string[index]

    def __str__(self):
        return self.string

    def __repr__(self):
        return '<%s at 0x%x (axiom=%s, rules=%s)' % \
               (self.__class__.__name__, id(self), self.axiom, \
                '/'.join(['%s:%s' % x for x in self.rules]))


class SequentialLSystem(LSystem):

    """An alternative L-system where at most one transition is made
    per generation; each rule's input is checked in order to see if it
    matches anywhere, and if so, only the leftmost transition is
    made."""
    
    def step(self):
        self.generation += 1
        for input, output in self.rules:
            index = self.string.find(input)
            if index >= 0:
                indexEnd = index + len(input)
                self.string = self.string[:index] + output + \
                              self.string[indexEnd:]
                return
        self.done = 1

class TagLSystem(LSystem):

    """Another alternative L-system, again where at most one
    transition is made per generation; each rule's input is checked in
    order only at the beginning of the string, and if it matches, the
    output is substituted at the end of the string."""

    def step(self):
        self.generation += 1
        for input, output in self.rules:
            if self.string.startswith(input):
                self.string = self.string[len(input):] + output
                return
        self.done = 1


def main():
    systems = [LSystem('A', [('A', 'AB'), ('B', 'BA')]), 
               SequentialLSystem('BABA', [('ABA', 'AAB'), ('A', 'ABA')]),
               TagLSystem('BB', [('BB', 'AAA'), ('BA', ''), ('AB', 'BA'), ('AA', 'BB')])]
    for system in systems:
        print(repr(system))
        while not system.done and system.generation < 10:
            print(str(system.generation) + " " + str(system.string))
            system.step()

if __name__ == '__main__': main()
