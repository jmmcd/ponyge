#!/usr/local/bin/python

"""
Membrane computing system (P-system) in Python.

Based on LSystem by Erik Max Francis.
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
        self.done = 0

    def membranes_step(self):
        for membrane in self.membranes:
            membrane.step()

    def rules_step(self):
        i = 0
        new_string = ''
        string_len = len(self.string)
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

    def step(self):
        if self.membranes is not None:
            self.membranes_step()
        if self.rules is not None:
            self.rules_step()

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
        str =  depth_str + '%s\n' % self.string
        if (self.rules is not None):
            str += '\n'.join([depth_str + '%s -> %s' % x for x in self.rules])
        if (self.membranes is not None):
            str += '\n'.join([(  depth_str + '{\n' 
                               + depth_str + '%s\n' 
                               + depth_str + '}') 
                              % x.__repr__(depth) for x in self.membranes])
        return str


class PSystem(Membrane):
    """P-System."""
    
    def __init__(self, axiom, membranes=None):
        """Initialize P-system"""
        self.generation = 0
        Membrane.__init__(self, axiom, None, membranes)

    def step(self):
        """Execute one generation of the system."""
        self.generation += 1
        Membrane.step(self)


def main():
    r1 = ('A','AB')
    r2 = ('B','BAC')
    r3 = ('C','ABC')
    a1 = 'A'
    a2 = 'AB'
    m3 = Membrane(a1, [r1,r2])
    m2 = Membrane(a2, [r2,r3], [m3])
    m1 = Membrane(a1, [r1,r2])
    p = PSystem('A', [m1,m2])
    while p.generation < 10:
        print ('GENERATION: ' + str(p.generation) + '\n')
        print ('OUTPUT: ' + str(p.string) + '\n')
        print "PSYSTEM: \n"
        print p.__repr__()
        p.step()

if __name__ == '__main__': 
    main()
