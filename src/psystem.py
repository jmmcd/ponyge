#!/usr/local/bin/python

"""
Membrane computing system (P-system) in Python.
"""

__program__ 	= 'psystem'
__version__ 	= '1.0'
__author__  	= 'Mark Wagy <mark.wagy@gmail.com>'
__copyright__	= 'Copyright (c) 2011 Mark Wagy'
__license__ 	= 'GPL'


class Membrane:
    """Membrane."""
    axiom = None
    rules = None
    membranes = None

    def __init__(self, axiom=None, rules=None, membranes=None):
        """Initialize membrane"""
        if axiom is not None:
            self.axiom = axiom
        if rules is not None:
            self.rules = rules
        if membranes is not None:
            self.membranes = membranes
        self.string = self.axiom

    def step(self):
        for membrane in membranes:
            while i < string_len:
                for input, output in self.rules:
                    if self.string[i:i+len(input)] == input:
                        new_string += output
                    i += len(input)
                    break
                else:
                    new_string += self.string[i]
                    i += 1
                if self.string == new_string:
                    self.done = 1
                self.string = new_string

    def reset(self):
        """Reset the system to its axiom"""
        self.string = self.axiom

    def __len__(self):
        return len(self.string)

    def __getitem__(self, index):
        return self.string[index]
    
    def __str__(self):
        return self.string

    def get_rules(self):
        return self.rules

    def __repr__(self, depth=None):
        # indent nested membranes
        if depth is not None:
            depth += 1
        else:
            depth = 0
        depth_str = ''
        for i in range(0,depth):
            depth_str += ' '
        str =  depth_str + '%s\n' % self.axiom
        if (self.rules is not None):
            str += '\n'.join([depth_str + '%s -> %s' % x for x in self.rules])
        if (self.membranes is not None):
            str += '\n'.join(['\n' + depth_str + '{\n%s\n}' 
                              % x.__repr__(depth) for x in self.membranes])
        return str


class PSystem:
    """P-System."""

    output_axiom = None
    membranes = None

    def __init__(self, membranes=None):
        """Initialize P-system"""
        membranes = None
        if (membranes is not None):
            self.membranes = membranes
        self.generation = 0
        self.done = 0

    def step(self):
        """Execute one generation of the system."""
        self.generation += 1
        for membrane in membranes:
            membrane.step()

def main():
    r1 = ('A','AB')
    r2 = ('B','BA')
    r3 = ('C','ABC')
    a1 = 'A'
    a2 = 'AB'
    m3 = Membrane(a1, [r1,r2])
    m2 = Membrane(a2, [r2,r3], [m3])
    m1 = Membrane(a1, [r1,r2])
    p1 = PSystem([m1,m2])

if __name__ == '__main__': 
    main()


r1 = ('A','AB')
r2 = ('B','BA')
r3 = ('C','ABC')
a1 = 'A'
a2 = 'AB'
m3 = Membrane(a1, [r1,r2])
m2 = Membrane(a2, [r2,r3], [m3])
m1 = Membrane(a1, [r1,r2])
p1 = PSystem([m1,m2])
