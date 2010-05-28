#! /usr/bin/env python

# Draw an lsystem

#TODO Color
#TODO Scale
#TODO Line Width

import turtle
import lsystem
import re
import math

class Drawing(turtle.Turtle):
    """Class for drawing"""

    def f(self): #forward no drawing
        self.penup()
        self.forward(self.step)

    def F(self): #forward drawing
        self.pendown()
        self.forward(self.step)
        self.penup()

    def C(self): #drawing an arc with radius step and angle circle_angle
        self.pendown()
        self.circle(self.step, self.circle_angle)
        self.penup()

    def D(self): #drawing a dot of default size
        self.dot()

    def r(self): #turning right
        self.right(self.angle)

    def l(self): #turning left
        self.left(self.angle)

    def S(self): #increase step size
        self.step += self.STEP

    def s(self): #decrease step size
        self.step -= self.STEP

    def P(self): #begin polygon
        self.begin_poly()
        self.begin_fill()

    def p(self): #end polygon
        self.end_poly()
        self.end_fill()

    def a(self): #decrease angle
        self.angle -= self.ANGLE

    def A(self): #increase angle
        self.angle += self.ANGLE

    def X(self): #Do nothing
        pass

    def n(self): # increase pen palette parameter
        self.pen_colour += 0.2
        self.pencolor(self.map_colour(self.pen_colour))

    def m(self): # decrease pen palette parameter
        self.pen_colour -= 0.2
        self.pencolor(self.map_colour(self.pen_colour))

    def N(self): # increase fill palette parameter
        self.fill_colour += 0.2
        self.fillcolor(self.map_colour(self.fill_colour))

    def M(self): # decrease fill palette parameter
        self.fill_colour -= 0.2
        self.fillcolor(self.map_colour(self.fill_colour))

    def _push(self): #push the (position, heading) to the stack
        self.stack.append((self.position(),self.heading()))

    def _pop(self): #pop and set the (position, heading) from the stack
        item = self.stack.pop()
        self.set_state(item)

    def __init__(self, l_system, depth, max_length=None):
        """Set the lsystem and the initial parameters"""
        super(Drawing,self).__init__()
        self.l_system = l_system
        self.depth = depth
        self.STEP = 2
        self.ANGLE = 5
        self.step = 10
        self.pen_colour = 0.0
        self.fill_colour = 0.0
        self.circle_angle = 20.5
        self.angle = 60
        self.max_length = max_length
        #Avalible rules
        self._rules = {"-":self.l, 
                       "+":self.r, 
                       "f":self.f, 
                       "F":self.F, 
                       "[":self._push, 
                       "]":self._pop, 
                       "C":self.C, 
                       "S":self.S, 
                       "s":self.s, 
                       "X":self.X, 
                       "A":self.a, 
                       "D":self.D, 
                       "}":self.p,
                       "{":self.P,
                       "a":self.a,
                       "n":self.n,
                       "m":self.m,
                       "N":self.N,
                       "M":self.M}
        self.stack = []
        self.colors = []

    def draw(self, x, y, width, heigth):
        """Draw the string. The l-system axiom is extended to the specified depth"""
        self.reset()
        turtle.setup(width,heigth,None,None)
        turtle.tracer(200,0)
        self.penup()
        self.setposition(x,y)
        while not self.l_system.done and self.l_system.generation < self.depth:
            self.l_system.step()
            if (self.max_length is not None and 
                len(self.l_system.string) > self.max_length):
                print("Exceeded maximum length: will not draw L-system")
                self.hideturtle()
                return
        print(self.l_system.string)
        self._draw(self.l_system.string, self._rules)
        self.hideturtle()
        turtle.update()

    def _draw(self, commands, rules):
        """Call the function in the command specified by the passed in rules"""
        for b in commands:
            try:
                rules[b]()
            except TypeError:
                try:
                    self._draw(rules[b], rules)
                except:
                    print("Passing:",b,rules[b])

    def set_state(self, state):
        self.setposition(state["position"])
        self.setheading(state["heading"])
        self.pen_colour = state["pen_colour"]
        self.fill_colour = state["fill_colour"]

    def make_state(self):
        d = {}
        d["position"] = self.position()
        d["heading"] = self.heading()
        d["pen_colour"] = self.pen_colour
        d["fill_colour"] = self.fill_colour
        return d

    # We use a similar palette scheme to that of Hart (EvoMUSART
    # 2007). There are two "anchor colours", set as constants in the
    # input string. There is a single palette parameter. Large
    # positive values give a saturated version of colour1 (tending to
    # white), small positive values a dark version tending to black.
    # Negative values work similarly for colour2.
    def map_colour(self, col):
        print("col = " + str(col))
        # Sigmoid maps values from [0, inf] to [0, 1]
        def sigmoid(val):
            return 1.0 / (1 + math.exp(val * abs(col) / 255.0))
        if col < 0:
            return tuple(map(sigmoid, self.colour2))
        else:
            return tuple(map(sigmoid, self.colour1))

def dragon_curve(depth=3):
    """draw a dragon curve Dragon curve angle=60"""
    _lsystem = lsystem.LSystem('F',[('F','F-F++F-F')])
    _drawing = Drawing(_lsystem, depth)
    _drawing.angle = 60
    _drawing.step = 20
    _drawing.draw(0,0,1000,750)

def simple_branch(depth=2):
    """draw a simple branch Branch angle=22.5"""
    _lsystem = lsystem.LSystem('F',[('F','FF-[-F+F+F]+[+F-F-F]')])
    _drawing = Drawing(_lsystem, depth)
    _drawing.angle = 22.5
    _drawing.step = 20
    _drawing.draw(0,0,1000,750)

def pyramid(depth=3):
    """Pyramid angle=60"""
    _lsystem = lsystem.LSystem('FXF--FF--FF',[('F','FF'),('X','--FXF++FXF++FXF--')])
    _drawing = Drawing(_lsystem, depth)
    _drawing.angle = 60
    _drawing.step = 20
    _drawing.draw(0,0,1000,750)

def curve_branch(depth=3):
    """Branch Curve angle=22.5, step=40, circle angle=20.5, STEP=2"""
    _lsystem = lsystem.LSystem('C+C',[('C','CC-[-C+C+C]+[-SC-C-sC]')])
    _drawing = Drawing(_lsystem, depth)
    _drawing.angle = 22.5
    _drawing.step = 40
    _drawing.circle_angle = 20.5
    _drawing.STEP = 2
    _drawing.draw(0,0,1000,750)

def parse_phenotype(phenotype):
    """Parses a phenotype: (The keys for the rules are allowed in the
    axiom or rules)
    angle=[\d]*\.?[\d]+ 
    depth=\d+ 
    step_size=\d+
    colour1=\d+ \d+ \d+
    colour2=\d+ \d+ \d+
    circle_angle=[\d]*\.?[\d]+ 
    axiom=[-+fF\[\]CSsXAD\{\}a]+
    [sSaADfFCX]=[-+fF\[\]CSsXAD\{\}a]+"""
    #TODO can I get the keys for the rules allowed by drawing
    REGEX_FLOAT = '\d+\.?\d+'
    REGEX_INTEGER = '\d+'
    REGEX_3_INTEGER = '(\d+) (\d+) (\d+)'
    REGEX_RULE_KEYS = '[-+fF\[\]CSsXADnmNM\{\}a]+'
    REGEX_RULE = '^(%s)=(%s)$'%(REGEX_RULE_KEYS,REGEX_RULE_KEYS)
    REGEX_AXIOM = 'axiom=(%s)'%(REGEX_RULE_KEYS)
    lines = phenotype.split(':')
    print(lines)
    p_dict = {}
    p_dict['angle'] = float(re.search(REGEX_FLOAT, lines[0]).group(0))
    p_dict['depth'] = int(re.search(REGEX_INTEGER, lines[1]).group(0))
    p_dict['step_size'] = int(re.search(REGEX_INTEGER, lines[2]).group(0))
    p_dict['colour1'] = tuple(map(int, re.search(REGEX_3_INTEGER, lines[3]).group(1, 2, 3)))
    p_dict['colour2'] = tuple(map(int, re.search(REGEX_3_INTEGER, lines[4]).group(1, 2, 3)))
    p_dict['circle_angle'] = float(re.search(REGEX_FLOAT, lines[5]).group(0))
    p_dict['axiom'] = re.search(REGEX_AXIOM,lines[6]).group(1)
    rules = []
    for line in lines[7:]:
        match = re.search(REGEX_RULE, line)
        if match is not None:
            rules.append((match.group(1), match.group(2)))
    p_dict['rules'] = rules
    print(p_dict)
    return p_dict

if __name__ == "__main__":
#Used for doodling drawings
#Spirograph
#    _lsystem = lsystem.LSystem('DC',[('C','CaD++[sCDsCD]++CaD'),('F','')])
    phenotype = 'angle=60:depth=2:step_size=10:colour1=200 50 50:colour2=50 200 50:circle_angle=20.5:axiom=F:F=mmF-F++F-F'
    p_dict = parse_phenotype(phenotype)
    _lsystem = lsystem.LSystem(p_dict['axiom'],p_dict['rules'])
    _drawing = Drawing(_lsystem, p_dict['depth'])
    _drawing.angle = p_dict['angle']
    _drawing.step = p_dict['step_size']
    _drawing.colour1 = p_dict['colour1']
    _drawing.colour2 = p_dict['colour2']
    _drawing.circle_angle = p_dict['circle_angle']
    _drawing.STEP = 2
    _drawing.ANGLE = 5
    _drawing.draw(0,0,1000,750)
#    dragon_curve()
    while True:
        from time import sleep
        sleep(3)

