 #! /usr/bin/env python

# Draw an lsystem

#TODO Color
#TODO Scale
#TODO Line Width

import turtle
import lsystem

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

    def a(self): #decrease angle
        self.angle -= self.ANGLE

    def A(self): #increase angle
        self.angle += self.ANGLE

    def X(self): #Do nothing
        pass

    def _push(self): #push the (position, heading) to the stack
        self.stack.append((self.position(),self.heading()))

    def _pop(self): #pop and set the (position, heading) from the stack
        item = self.stack.pop()
        self.setposition(item[0])
        self.setheading(item[1])

    def __init__(self, l_system, depth):
        """Set the lsystem and the initial parameters"""
        super(Drawing,self).__init__()
        self.l_system = l_system
        self.depth = depth
        self.STEP = 2
        self.ANGLE = 5
        self.step = 40
        self.circle_angle = 20.5
        self.angle = 22.5
        #Avalible rules
        self._rules = {"-":self.r, 
                       "+":self.l, 
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
                       "a":self.a}
        self.stack = []

    def draw(self, x, y, width, heigth):
        """Draw the string. The l-system axiom is extended to the specified depth"""
        self.reset()
        turtle.setup(width,heigth,None,None)
        turtle.tracer(200,0)
        self.setposition(x,y)
        while not self.l_system.done and self.l_system.generation < self.depth:
            self.l_system.step()
        print(self.l_system.string)
        self._draw(self.l_system.string, self._rules)
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

if __name__ == "__main__":
#Used for doodling drawings
#Spirograph
    _lsystem = lsystem.LSystem('DC',[('C','CaD++[sCDsCD]++CaD'),('F','')])
    _drawing = Drawing(_lsystem, 2)
    _drawing.angle = 22.5
    _drawing.step = 20
    _drawing.circle_angle = 90
    _drawing.STEP = 2
    _drawing.ANGLE = 5
    _drawing.draw(0,0,1000,750)
    while True:
        from time import sleep
        sleep(3)

