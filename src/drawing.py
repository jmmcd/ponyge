#! /usr/bin/env python

# Draw an lsystem

#TODO Scale -- not needed since saves are in .eps?
#TODO Line Width
#TODO Rename "redisplay" to "back" and allow it to go to previous generations as well.
#TODO Reverse an individual's string? (turn brackets around)
#TODO Alter an individual's palette
#TODO Angle steps should be multiples of initial angle?

# FIXME When saving postscript, all the graphics are saved, but the
# bounding box is set to the visible window. It's possible to edit the
# .eps to fix the bounding box and see the entire picture. But it
# would be better to set options to the postscript call to set the
# bounding box correctly before writing.

# FIXME Change the elitism -- seems one individual is always
# reproduced but doesn't contribute genes?

import turtle
import lsystem
import re
import math
import sys
import time

class Attractor:
    """Attractor or repulsor"""
    #TODO currently only checked in forward movement, check when drawing circle?
    def __init__(self, type_, effect, xcor, ycor, side):
        """Type, effect center coordinate and sides"""
        #TODO make the force field a different shape than a square?
        self.type_ = type_
        self.effect = effect
        self.xcor = (xcor - side, xcor + side)
        self.ycor = (ycor - side, ycor + side)
        self.side = side
        print('%s __init__ type:%s effect:%s x_min:%.3f x_max:%.3f y_min:%.3f y_max:%.3f' % (__name__, self.type_, self.effect, self.xcor[0], self.xcor[1], self.ycor[0], self.ycor[1]))

    def force_field_effect(self, turtle):
        """Find turtle's position relative to its origin. Check if
        it's in the force field, then apply force field effect if
        so."""
        relpos_x = turtle.xcor() - self.origin[0]
        relpos_y = turtle.ycor() - self.origin[1]

        if self.xcor[0] < relpos_x < self.xcor[1] and \
                self.ycor[0] < relpos_y < self.ycor[1]:
            if self.type_ == 'positive':
                if self.effect == 'gravity':
                    turtle.S()
                elif self.effect == 'color':
                    turtle.n()
            else:
                if self.effect == 'gravity':
                    turtle.s()
                elif self.effect == 'color':
                    turtle.m()

class Drawing(turtle.Turtle):
    """Class for drawing"""

    def force_field(self):
        """Check if force_fields affects the turtle"""
        if self.force_fields:
            for force_field in self.force_fields:
                force_field.force_field_effect(self)

    def f(self):
        """Forward no drawing"""
        self.penup()
        self.forward(self.step)
        self.force_field()

    def F(self):
        """Forward drawing"""
        self.pendown()
        self.forward(self.step)
        self.penup()
        self.force_field()

    def C(self):
        """Drawing an arc with radius step and angle circle_angle"""
        self.pendown()
        self.circle(self.step, self.circle_angle)
        self.penup()

    def D(self):
        """Drawing a dot of default size"""
        self.dot()

    def r(self):
        """Turning right"""
        self.right(self.angle)

    def l(self):
        """Turning left"""
        self.left(self.angle)

    def S(self):
        """Increase step size"""
        self.step += self.STEP
        if self.step >= 20:
            self.step = 20

    def s(self):
        """Decrease step size"""
        self.step -= self.STEP
        if self.step <= 2:
            self.step = 2

    def polygon_begin(self):
        """Begin polygon"""
        self.begin_poly()
        self.begin_fill()

    def polygon_end(self):
        """End polygon"""
        self.end_poly()
        self.end_fill()

    def a(self):
        """Decrease angle"""
        self.ANGLE -= 1
        self.ANGLE %= len(self.set_angles)
        self.angle = self.set_angles[self.ANGLE]

    def A(self):
        """Increase angle"""
        self.ANGLE += 1
        self.ANGLE %= len(self.set_angles)
        self.angle += self.set_angles[self.ANGLE]

    def X(self):
        """Do nothing"""
        pass

    def n(self):
        """Increase pen palette parameter"""
        self.pen_colour += 1
        self.pencolor(self.map_colour(self.pen_colour))

    def m(self):
        """Decrease pen palette parameter"""
        self.pen_colour -= 1
        self.pencolor(self.map_colour(self.pen_colour))

    def N(self):
        """Increase fill palette parameter"""
        self.fill_colour += 1
        self.fillcolor(self.map_colour(self.fill_colour))

    def M(self):
        """Decrease fill palette parameter"""
        self.fill_colour -= 1
        self.fillcolor(self.map_colour(self.fill_colour))

    def w(self):
        """Decrease pen width"""
        self.pen_width /= 1.05
        if self.pen_width < 0.5:
            self.pen_width = 0.5
        self.width(self.pen_width)

    def W(self):
        """Increase pen width"""
        self.pen_width *= 1.05
        if self.pen_width > 3:
            self.pen_width = 3
        self.width(self.pen_width)

    def _push(self):
        """Push the (position, heading) to the stack"""
        self.stack.append(self.make_state())

    def _pop(self):
        """Pop and set the (position, heading) from the stack"""
        self.set_state(self.stack.pop())

    def __init__(self, grammar_system, depth,
                 step=10, angle=4, circle_angle=20.5,
                 colour1=(200, 0, 0), # red
                 colour2=(0, 200, 0), # green
                 STEP=2, ANGLE=5,
                 pen_width=3.0):
        """Set the lsystem and the initial parameters"""
        super(Drawing,self).__init__()
        self.grammar_system = grammar_system
        self.depth = depth
        self.STEP = STEP
        self.set_angles = [10, 12, 15, 20, 24, 27.5, 30, 360.0 / 11,
                           36.0, 40.0, 45, 360.0 / 7, 60, 72, 90]
        self.ANGLE = ANGLE
        self.step = step
        self.pen_colour = 0
        self.fill_colour = 0
        self.colour1 = colour1
        self.colour2 = colour2
        self.circle_angle = circle_angle
        self.angle = angle
        self.pen_width = pen_width
        #Available rules
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
                       "{":self.polygon_begin,
                       "}":self.polygon_end,
                       "A":self.A,
                       "D":self.D,
                       "a":self.a,
                       "n":self.n,
                       "m":self.m,
                       "N":self.N,
                       "M":self.M,
                       "w":self.w,
                       "W":self.W
                       }
        self.drawing_commands = "FCD}"
        self.stack = []
        self.force_fields = []

    # Return True if the phenotype does not exceed maximum length, and
    # drawing contains some commands which actually paint something.
    def draw(self, x, y, width, height, max_length=None, force_fields=None):
        """Draw the string. The grammar-system axiom is extended to
        the specified depth"""
        self.reset()
        turtle.setup(width,height,None,None)
        turtle.tracer(200,0)
        self.penup()
        self.setposition(x,y)
        self.origin = x, y
        self.max_length = max_length
        while not self.grammar_system.done and \
                self.grammar_system.generation < self.depth:
            self.grammar_system.step()
            if (self.max_length is not None and
                len(self.grammar_system.string) > self.max_length):
                self.hideturtle()
                print("Drawing exceeded maximum length")
                return False
        print(self.grammar_system.string)

        if force_fields:
            for force_field in force_fields:
                self.force_fields.append(Attractor(force_field['type'], force_field['effect'], force_field['x'], force_field['y'], force_field['size']))

        non_null = self._draw(self.grammar_system.string, self._rules)
        self.hideturtle()
        turtle.update()
        return non_null

    # Return True if some of the commands actually draw something, as
    # opposed to just moving around.
    def _draw(self, commands, rules):
        """Call the function in the command specified by the passed in rules"""
        null_drawing = not any(d in commands for d in self.drawing_commands)
        if null_drawing:
            print("No drawing commands in drawing string")
            return False
        for b in commands:
            # Removed exception-handling -- if there's a bad command,
            # we want to know about it.
            rules[b]()
        return True

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

    # Palette scheme is circular:
    # 0 -> colour1
    # 1/3 -> colour2
    # 2/3 -> black
    # 1 -> colour1
    def map_colour(self, col):
        black = (0.0, 0.0, 0.0)
        granularity = 100
        v = (col % granularity) / float(granularity)
        # linear interpolation between a and b.
        def lin(a, b, t):
            return tuple((ai + t * (bi - ai)) / 256.0 for ai, bi in zip(a, b))
        if v < 1/3.0:
            return lin(self.colour1, self.colour2, v * 3.0)
        elif v < 2/3.0:
            return lin(self.colour2, black, (v - 1/3.0) * 3.0)
        else:
            return lin(black, self.colour1, (v - 2/3.0) * 3.0)

def dragon_curve(depth=3):
    """draw a dragon curve Dragon curve angle=60"""
    _lsystem = lsystem.LSystem('F',[('F','F-F++F-F')])
    _drawing = Drawing(_lsystem, depth, angle=60, step=20)
    _drawing.draw(0,0,1000,750)

def simple_branch(depth=2):
    """draw a simple branch Branch angle=22.5"""
    _lsystem = lsystem.LSystem('F',[('F','FF-[-F+F+F]+[+F-F-F]')])
    _drawing = Drawing(_lsystem, depth, angle=22.5, step=20)
    _drawing.draw(0,0,1000,750)

def pyramid(depth=3):
    """Pyramid angle=60"""
    _lsystem = lsystem.LSystem('FXF--FF--FF',[('F','FF'),('X','--FXF++FXF++FXF--')])
    _drawing = Drawing(_lsystem, depth, angle=60, step=20)
    _drawing.draw(0,0,1000,750)

def curve_branch(depth=3):
    """Branch Curve angle=22.5, step=40, circle angle=20.5, STEP=2"""
    _lsystem = lsystem.LSystem('C+C',[('C','CC-[-C+C+C]+[-SC-C-sC]')])
    _drawing = Drawing(_lsystem, depth, angle=22.5, step=40, circle_angle=20.5, STEP=2)
    _drawing.draw(0,0,1000,750)

def six_pointed_star(depth=3):
    """A nice 6-pointed angled star"""
    _lsystem = lsystem.LSystem("F", [("F", "m[F-F++F-F]")])
    _drawing = Drawing(_lsystem, 4, angle=60, step=50,
                       colour1=(200, 50, 50), colour2=(50, 200, 50),
                       circle_angle=20.5, STEP=2, ANGLE=5)
    _drawing.draw(0,0,1000,750)

def koch(depth=3):
    """Simplest L-system example."""
    _lsystem = lsystem.LSystem("F", [("F", "F-F++F-F")])
    _drawing = Drawing(_lsystem, depth, angle=60, step=100.0 / (depth ** 2.0),
                      colour1=(200, 50, 50), colour2=(50, 200, 50),
                      circle_angle=20.5, STEP=2, ANGLE=5)
    _drawing.draw(0,0,1000,750)

def doodle():
    # _lsystem is basically ignored
    _lsystem = lsystem.LSystem('f',[('f','f')])
    _drawing = Drawing(_lsystem,
                       1, angle=90,
                       step=100,
                       colour1=(200, 50, 50),
                       colour2=(50, 200, 50),
                       circle_angle=20.5,
                       STEP=2, ANGLE=5)
    # we draw this hand-written string
    _drawing._draw("FF+F+FF+DF+D", _drawing._rules)


if __name__ == "__main__":

    # dragon_curve()
    # pyramid()
    # curve_branch()
    # simple_branch()
    # six_pointed_star()
    # koch(4)
    doodle()

    time.sleep(5)
