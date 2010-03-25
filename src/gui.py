#!/usr/bin/env python3

# GUI for interactive GE with PonyGE. Uses TK to create a simple GUI
# and turtle graphics to draw a grid of spaces for individuals.
# Individuals can then be drawn using turtle graphics also (or
# pre-rendered as image files, depending on their representation) and
# then stamped using turtle graphics.

# This GUI is based on the turtleDemo.py example which comes with the
# turtle module in python 2.6/3.

# Requires: Python 2.6/3; lsystem.py (from PyPI); ponyge.py;
# drawing.py (part of ponyge).

import sys
import os

from tkinter import *
from idlelib.Percolator import Percolator
from idlelib.ColorDelegator import ColorDelegator
from idlelib.textView import view_file # TextViewer
from imp import reload
import ponyge
import lsystem
import drawing

import turtle
import time

STARTUP = 1
READY = 2
RUNNING = 3
DONE = 4
EVENTDRIVEN = 5

INSTRUCTIONS = "Click individuals, then press next"

btnfont = ("Arial", 12, 'bold')
txtfont = ('Lucida Console', 8, 'normal')

class DummyFitness():
    maximise = True
    def __call__(self, candidate):
        return 0.0

class GE(object):
    def __init__(self):
        self.generation = 0
        ponyge.GRAMMAR_FILE = "grammars/letter.bnf"
        ponyge.POPULATION_SIZE = 9
        ponyge.GENERATION_SIZE = 9
        self.grammar = ponyge.Grammar(ponyge.GRAMMAR_FILE)
        self.individuals = ponyge.initialise_population(ponyge.POPULATION_SIZE)
        self.replacement = ponyge.generational_replacement
        self.selection = ponyge.tournament_selection
        self.fitness_function = DummyFitness()
        ponyge.evaluate_fitness(self.individuals, self.grammar, self.fitness_function)
        self.best_ever = min(self.individuals)
        self.individuals.sort()
        ponyge.print_stats(1, self.individuals)
    def step(self):
        print("nindividuals: " + str(len(self.individuals)) + "; fitnesses:")
        print(str(list(map(lambda x: x.fitness, self.individuals))))

        self.individuals, self.best_ever = ponyge.step(
            self.individuals, self.grammar, self.replacement,
            self.selection, self.fitness_function, self.best_ever)
        ponyge.print_stats(self.generation, self.individuals)
        self.generation += 1

    def set_fitnesses(self, fitnesses):
        for ind, fitness in zip(self.individuals, fitnesses):
            ind.fitness = fitness


class MyTurtle(object):
    # width and height in pixels. n and m specify grid. n * m = populationsize
    def __init__(self, width, height, n, m):
        self.n = n
        self.m = m
        self.margin = 10
        self.width = width
        self.height = height
        self.xside = width / float(n)
        self.yside = height / float(m)
        self.xside_box = self.xside - self.margin * 2
        self.yside_box = self.yside - self.margin * 2

    def index_to_pixel(self, idx, axis="x"):
        if axis == "x":
            return -self.width / 2.0 + idx * self.xside + self.margin
        else:
            return -self.height / 2.0 + idx * self.yside + self.margin
    def pixel_to_index(self, pixel, axis="x"):
        if axis == "x":
            return int(pixel + self.width / 2.0) // int(self.xside)
        else:
            return int(pixel + self.height / 2.0) // int(self.yside)

    def drawFrame(self, i, j, colour=None):
        if colour is not None:
            turtle.color("red")
        else:
            turtle.color("black")
        turtle.up()
        turtle.tracer(100,0)
        x = self.index_to_pixel(i, "x")
        y = self.index_to_pixel(j, "y")
        print("going to " + str(x) + " " + str(y))
        turtle.setx(x)
        turtle.sety(y)
        turtle.down()
        turtle.forward(self.xside_box)
        turtle.left(90)
        turtle.forward(self.yside_box)
        turtle.left(90)
        turtle.forward(self.xside_box)
        turtle.left(90)
        turtle.forward(self.yside_box)
        turtle.left(90)
        turtle.update()
    def get_i_j(self, x, y):
        return (self.pixel_to_index(x, "x"), self.pixel_to_index(y, "y"))

class GUI(object):

    def __init__(self, filename=None):   #, root=None):
        self.root = root = turtle._root = Tk()
        root.wm_protocol("WM_DELETE_WINDOW", self._destroy)

        #################
        self.mBar = Frame(root, relief=RAISED, borderwidth=2)
        self.mBar.pack(fill=X)

        root.title('PonyGE GUI')
        #################
        self.left_frame = left_frame = Frame(root)
        self.text_frame = text_frame = Frame(left_frame)
        self.vbar = vbar =Scrollbar(text_frame, name='vbar')
        self.text = text = Text(text_frame,
                                name='text', padx=5, wrap='none',
                                width=45)
        vbar['command'] = text.yview
        vbar.pack(side=LEFT, fill=Y)
        #####################
        self.hbar = hbar =Scrollbar(text_frame, name='hbar', orient=HORIZONTAL)
        hbar['command'] = text.xview
        hbar.pack(side=BOTTOM, fill=X)
        #####################
        text['yscrollcommand'] = vbar.set
        text.config(font=txtfont)
        text.config(xscrollcommand=hbar.set)
        text.pack(side=LEFT, fill=Y, expand=1)
        #####################
        self.output_lbl = Label(left_frame, height= 1,text=" --- ", bg = "#ddf",
                                font = ("Arial", 16, 'normal'))
        self.output_lbl.pack(side=BOTTOM, expand=0, fill=X)
        #####################
        text_frame.pack(side=LEFT, fill=BOTH, expand=0)
        left_frame.pack(side=LEFT, fill=BOTH, expand=0)
        self.graph_frame = g_frame = Frame(root)

        turtle._Screen._root = g_frame
        turtle._Screen._canvas = turtle.ScrolledCanvas(g_frame, 600, 600, 600, 600)
        #xturtle.Screen._canvas.pack(expand=1, fill="both")
        self.screen = _s_ = turtle.Screen()
#####
        turtle.TurtleScreen.__init__(_s_, _s_._canvas)
#####
        self.scanvas = _s_._canvas
        #xturtle.RawTurtle.canvases = [self.scanvas]
        turtle.RawTurtle.screens = [_s_]

        self.scanvas.pack(side=TOP, fill=BOTH, expand=1)

        self.btn_frame = btn_frame = Frame(g_frame, height=100)
        self.next_btn = Button(btn_frame, text=" NEXT ", font=btnfont, fg = "white",
                                disabledforeground = "#fed", command=self.nextGeneration)
        self.next_btn.pack(side=LEFT, fill=X, expand=1)
        self.stop_btn = Button(btn_frame, text=" STOP ",  font=btnfont, fg = "white",
                                disabledforeground = "#fed", command = self.stopIt)
        self.stop_btn.pack(side=LEFT, fill=X, expand=1)

        self.btn_frame.pack(side=TOP, fill=BOTH, expand=0)
        self.graph_frame.pack(side=TOP, fill=BOTH, expand=1)

        # Grid size, giving population
        self.n = 3
        self.m = 3
        # Set up PonyGE
        self.ge = GE()
        self.fitness = [0.0 for i in range(self.n) for j in range(self.m)]

        Percolator(text).insertfilter(ColorDelegator())
        self.dirty = False
        self.exitflag = False
        self.configGUI(NORMAL, DISABLED, DISABLED, DISABLED, INSTRUCTIONS)
        self.state = STARTUP
        self.nextGeneration()

    def clickcb(self, x, y):
        print("self.clickcb: x = " + str(x) + " " + str(y))
        ij = self.myt.get_i_j(x, y)
        print(str(ij))
        self.setSelected(*ij)

    def _destroy(self):
        self.root.destroy()
        sys.exit()

    def configGUI(self, menu, next, stop, clear, txt="", color="blue"):

        self.next_btn.config(state=next)
        if next==NORMAL:
            self.next_btn.config(bg="#d00")
        else:
            self.next_btn.config(bg="#fca")

        self.stop_btn.config(state=stop)
        if stop==NORMAL:
            self.stop_btn.config(bg="#d00")
        else:
            self.stop_btn.config(bg="#fca")

        self.output_lbl.config(text=txt, fg=color)


    def refreshCanvas(self):
        if not self.dirty: return
        self.screen.clear()
        #self.screen.mode("standard")
        self.dirty=False

    def nextGeneration(self):
        self.refreshCanvas()
        self.dirty = True
        turtle.TurtleScreen._RUNNING = True
        self.configGUI(DISABLED, DISABLED, NORMAL, DISABLED,
                       "demo running...", "black")
        self.screen.clear()
        self.screen.mode("standard")
        self.state = RUNNING

        self.selected = []
        self.configGUI(DISABLED, DISABLED, NORMAL, DISABLED,
                       "Drawing, please wait...", "red")

        self.ge.set_fitnesses(self.fitness)
        self.ge.step()
        self.myt = MyTurtle(650, 650, 3, 3)
        turtle.clear()
        for i in range(self.n):
            for j in range(self.m):
                self.setUnselected(i, j)
                #Drawing l-system                
                phenotype = 'angle=6%d\ndepth=%d\nstep_size=10\ncircle_angle=20.5\naxiom=F\nF=F-F++F-F'%((i*j),(i*j))
#                phenotype = self.ge.individuals[i*self.n+j].phenotype
                print(phenotype)
                p_dict = drawing.parse_phenotype(phenotype)
                _lsystem = lsystem.LSystem(p_dict['axiom'],p_dict['rules'])
                _drawing = drawing.Drawing(_lsystem, p_dict['depth'])
                _drawing.angle = p_dict['angle']
                _drawing.step = p_dict['step_size']
                _drawing.circle_angle = p_dict['circle_angle']
                _drawing.draw(self.myt.index_to_pixel(i, "x") + self.myt.xside_box / 2.0,
                              self.myt.index_to_pixel(j, "y") + self.myt.yside_box / 2.0,
                              self.myt.xside,
                              self.myt.yside)
        
        self.configGUI(NORMAL, NORMAL, DISABLED, NORMAL, INSTRUCTIONS)
        turtle.onscreenclick(self.clickcb, 1)

    def setSelected(self, i, j):
        if self.fitness[i*self.n + j] > 0.0:
            print("was selected, now de-selecting")
            self.setUnselected(i, j)
            return
        print("was unselected, now selecting")
        self.fitness[i*self.n + j] = 1.0
        self.myt.drawFrame(i, j, "red")

    def setUnselected(self, i, j):
        self.fitness[i*self.n + j] = 0.0
        self.myt.drawFrame(i, j)

    def clearCanvas(self):
        self.refreshCanvas()
        self.screen._delete("all")
        self.scanvas.config(cursor="")
        self.configGUI(NORMAL, NORMAL, DISABLED, DISABLED)

    def stopIt(self):
        if self.exitflag:
            self.clearCanvas()
            self.exitflag = False
            self.configGUI(NORMAL, NORMAL, DISABLED, DISABLED,
                           "STOPPED!", "red")
            turtle.TurtleScreen._RUNNING = False
            #print "stopIT: exitflag = True"
        else:
            turtle.TurtleScreen._RUNNING = False
            #print "stopIt: exitflag = False"

if __name__ == '__main__':
    _gui = GUI()
    RUN = True
    while RUN:
        try:
            print("ENTERING mainloop")
            _gui.root.mainloop()
        except AttributeError:
            print("AttributeError!- WAIT A MOMENT!")
            time.sleep(0.3)
            print("GOING ON ..")
            _gui.ckearCanvas()
        except TypeError:
            _gui.screen._delete("all")
            print("CRASH!!!- WAIT A MOMENT!")
            time.sleep(0.3)
            print("GOING ON ..")
            _gui.clearCanvas()
        except:
            print("BYE!")
            RUN = False

