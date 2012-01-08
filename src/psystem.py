#!/usr/local/bin/python

"""
Membrane computing system (P-system) in Python.

Built on LSystem code by Erik Max Francis.
"""

__program__ 	= 'psystem'
__version__ 	= '1.0'
__author__  	= 'Mark Wagy <mark.wagy@gmail.com>'
__copyright__	= 'Copyright (c) 2011 Mark Wagy'
__license__ 	= 'GPL'


# Membrane
# Represent idea of a cell membrane, modeled after the 
# biological sort, but not limited by that idea.
class Membrane:
    """Membrane."""
    axiom = None
    rules = None
    membranes = None
    id = 0
    dissolve = False

    # TODO: this is precarious, need a better idea of 'special' chars
    DISSOLVE_CHAR = "_"

    def __init__(self, axiom=None, rules=None, membranes=None, secrete_type=0):
        """Initialize membrane"""
        if axiom is not None:
            self.axiom = axiom
        if rules is not None:
            self.rules = rules
        if membranes is not None:
            self.membranes = membranes
        self.string = self.axiom
        self.done = 0
        self.secrete_type = secrete_type
        self.SPECIAL_CHARS = [self.DISSOLVE_CHAR]

    def membranes_step(self):
        for membrane in self.membranes:
            membrane.step()
            if membrane.dissolve:
                if membrane.rules is not None:
                    for rule in membrane.rules:
                        if self.rules is not None:
                            self.rules.append(rule)
                self.string += membrane.string
                self.membranes.remove(membrane)

    def rules_step(self):
        i = 0
        new_string = ''
        string_len = len(self.string)
        while i < string_len:
            for input, output in self.rules:
                if self.string[i:i+len(input)] == input:
                    new_string += output
                    i += len(input)
                    if output == self.DISSOLVE_CHAR:
                        self.dissolve = True
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

    def secrete(self):
        # default type: just return this membrane's string
        if (self.secrete_type == 0):
            return self.string
        # recursively (depth first) concatenate membrane strings
        elif (self.secrete_type == 1):
            full_string = ''
            if self.membranes is not None:
                for membrane in self.membranes:
                    mem_str = membrane.secrete()
                    if mem_str is not None:
                        full_string += mem_str
            full_string += self.string
            # move all special chars from secreted string
            return full_string.translate(None, self.get_special_chars_string())

    def get_special_chars_string(self):
        str = ''
        for c in self.SPECIAL_CHARS:
            str += c
        return str


class PSystem(Membrane):
    """P-System."""
    
    def __init__(self, axiom, membranes=None, secrete_type=0):
        """Initialize P-system"""
        self.generation = 0
        Membrane.__init__(self, axiom, None, membranes, secrete_type)

    def step(self):
        """Execute one generation of the system."""
        self.generation += 1
        Membrane.step(self)


def test_psystem1():
    r1 = ('A','ABY')
    r2 = ('B','BACY')
    r3 = ('C','ABCY')
    a1 = 'AZ'
    a2 = 'ABX'
    m3 = Membrane(a1, [r1,r2])
    m2 = Membrane(a2, [r2,r3], [m3])
    m1 = Membrane(a1, [r1,r2])
    psys = PSystem('A', [m1,m2], 1)
    return psys

# test membrane dissolve
def test_psystem2():
    r1 = ('a','b')
    r2 = ('b','_')
    s1 = 'a'
    s2 = 'b'
    m2 = Membrane(s2, [r2])
    m1 = Membrane(s1, [r1], [m2])
    psys = PSystem('z', [m1], 1)
    return psys

def test_psystem3():
    """Intended to be exactly the same as psystem2, but constructed in
    a single call."""
    psys = PSystem('z', [Membrane('a', [('a', 'b')], [
        Membrane('b', [('b', '_')])])], 1)
    return psys

def test_psystem4():
    """A randomly-generated one using the psystem.bnf grammar.
    Here's another random one:

    psys = PSystem("nNN", [Membrane("++mM{--}++{{FF+}}FFFnFNC[n+]nM", [("n", "[[F]]"),("C", "+m"),("m", "[FC]")], [])], 1)

    Can generate more like this:

    >>> import random; import ponyge
    >>> g = ponyge.Grammar("grammars/psystem.bnf")
    >>> genome = [random.randint(0, 100) for i in range(200)]
    >>> g.generate(genome)
    """

    psys = PSystem("F{M{FMM}}", [Membrane("n[[F]]+Fn", [("-", "+m"),("+", "+")], [Membrane("FCnF", [("M", "nF"),("F", "+N")], []),Membrane("C", [("-", "{-+}"),("-", "nF{F}")], [])]),Membrane("{[{[m]}]}", [("N", "MF")], [])], 1)

    return psys

def main():
    p = test_psystem2()
    while p.generation < 3:
        print ('====================\n')
        print ('GENERATION: ' + str(p.generation) + '\n')
        print ('OUTPUT: ' + str(p.secrete()) + '\n')
        print ("PSYSTEM: \n")
        print (p.__repr__())
        p.step()

if __name__ == '__main__': 
    main()
