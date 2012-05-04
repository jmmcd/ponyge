#!/usr/bin/env python

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
import time

try:
    from tkinter import *
except ImportError:
    from Tkinter import *
from idlelib.Percolator import Percolator
from idlelib.ColorDelegator import ColorDelegator
from idlelib.textView import view_file # TextViewer
from imp import reload
import ponyge
from lsystem import *
from psystem import *
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
    def __init__(self, grammar_file):
        self.generation = 0
        ponyge.GRAMMAR_FILE = grammar_file
        ponyge.POPULATION_SIZE = 9
        ponyge.GENERATION_SIZE = 9
        ponyge.ELITE_SIZE = 0
        ponyge.MUTATION_PROBABILITY = 0.01
        ponyge.CROSSOVER_PROBABILITY = 1.0 - ponyge.MUTATION_PROBABILITY
        self.grammar = ponyge.Grammar(ponyge.GRAMMAR_FILE)
        self.individuals = ponyge.initialise_population(ponyge.POPULATION_SIZE)
        self.replacement = ponyge.generational_replacement
        self.selection = lambda x: ponyge.tournament_selection(x, 6)
        ponyge.FITNESS_FUNCTION = DummyFitness()
        ponyge.evaluate_fitness(self.individuals, self.grammar, ponyge.FITNESS_FUNCTION)
        self.best_ever = min(self.individuals)
        self.individuals.sort()
        ponyge.print_stats(1, self.individuals)
    def step(self):
        print("nindividuals: " + str(len(self.individuals)) + "; fitnesses:")
        print(str(list(map(lambda x: x.fitness, self.individuals))))

        self.individuals, self.best_ever = ponyge.step(
            self.individuals, self.grammar, self.replacement,
            self.selection, ponyge.FITNESS_FUNCTION, self.best_ever)
        ponyge.print_stats(self.generation, self.individuals)
        # write to stderr. this allows us to redirect entire run to stdout
        # but still know which generation we're on, so can rename screenshots
        # correctly
        sys.stderr.write("Gen: " + str(self.generation) + "\n")
        self.generation += 1

    def print_to_file(self, file_name='tmp'):
        #TODO pythonify
        outfile = open(file_name+'.pop','w')
        outfile.write("generation: "+str(self.generation)+"; nindividuals: " + str(len(self.individuals)) + "; indifiduals:\n")
        for individual in self.individuals:
            outfile.write(str(individual)+'\n')
        outfile.close()

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
            assert(0 <= idx < self.n)
            return -self.width / 2.0 + idx * self.xside + self.margin
        else:
            assert(0 <= idx < self.m)
            return -self.height / 2.0 + idx * self.yside + self.margin

    # Clamp the value to a legal value
    def pixel_to_index(self, pixel, axis="x"):
        if axis == "x":
            result = int(pixel + self.width / 2.0) // int(self.xside)
            result = min(self.n - 1, result)
            result = max(0, result)
        else:
            result = int(pixel + self.height / 2.0) // int(self.yside)
            result = min(self.m - 1, result)
            result = max(0, result)
        return result

    def drawFrame(self, i, j, colour=None):
        if colour is not None:
            turtle.color("red")
        else:
            turtle.color("black")
        turtle.ht()
        turtle.up()
        turtle.tracer(10000,0)
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
        root.resizable(width=FALSE, height=FALSE)

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
        turtle._Screen._canvas = turtle.ScrolledCanvas(g_frame, 700, 700, 700, 700)
        #xturtle.Screen._canvas.pack(expand=1, fill="both")
        self.screen = _s_ = turtle.Screen()
#####
        turtle.TurtleScreen.__init__(_s_, _s_._canvas)
#####
        self.scanvas = _s_._canvas
        #xturtle.RawTurtle.canvases = [self.scanvas]
        turtle.RawTurtle.screens = [_s_]

        turtle.ht()

        self.scanvas.pack(side=TOP, fill=BOTH, expand=1)

        self.btn_frame = btn_frame = Frame(g_frame, height=100)
        self.next_btn = Button(btn_frame, text=" NEXT ", font=btnfont, fg = "white",
                                disabledforeground = "#fed", command=self.nextGeneration)
        self.next_btn.pack(side=LEFT, fill=X, expand=1)
        self.save_btn = Button(btn_frame, text=" SAVE ", font=btnfont, fg = "white",
                                disabledforeground = "#fed", command=self.savecb)
        self.save_btn.pack(side=LEFT, fill=X, expand=1)
        self.redisplay_btn = Button(btn_frame, text=" REDISPLAY ", font=btnfont, fg = "white",
                                disabledforeground = "#fed", command=self.redisplaycb)
        self.redisplay_btn.pack(side=LEFT, fill=X, expand=1)
        self.stop_btn = Button(btn_frame, text=" STOP ",  font=btnfont, fg = "white",
                                disabledforeground = "#fed", command = self.stopIt)
        self.stop_btn.pack(side=LEFT, fill=X, expand=1)

        self.btn_frame.pack(side=TOP, fill=BOTH, expand=0)
        self.graph_frame.pack(side=TOP, fill=BOTH, expand=1)

        # Grid size, giving population
        self.n = 3
        self.m = 3
        # Set up PonyGE
        self.ge = GE(GRAMMAR_FILE)
        self.fitness = [0.0 for i in range(self.n) for j in range(self.m)]

        Percolator(text).insertfilter(ColorDelegator())
        self.dirty = False
        self.exitflag = False
        self.configGUI(NORMAL, DISABLED, DISABLED, DISABLED, DISABLED, DISABLED, INSTRUCTIONS)

        # bring the window to front
        root.attributes('-topmost', 1)
        root.attributes('-topmost', 0)
        # none of these succeeds in bringing to front
        # root.lift()
        # root.tkraise()
        # none of these succeeds in focussing
        # root.focus()
        # root.focus_force()
        # self.next_btn.focus_force()

        self.state = STARTUP
        self.nextGeneration()

    def clickcb(self, x, y):
        print("self.clickcb: x = " + str(x) + " " + str(y))
        ij = self.myt.get_i_j(x, y)
        print(str(ij))
        self.setSelected(*ij)

    # clear screen, draw individual(i, j) again in the centre
    def rightclickcb(self, x, y):
        print("self.rightclickcb: x = " + str(x) + " " + str(y))
        ij = self.myt.get_i_j(x, y)
        print(str(ij))
        i, j = ij
        self.refreshCanvas()
        self.dirty = True
        turtle.TurtleScreen._RUNNING = True
        self.screen.clear()
        self.screen.mode("standard")
        self.state = RUNNING
        phenotype = self.ge.individuals[i*self.n+j].phenotype
        self.draw_phenotype(phenotype,
                            self.myt.index_to_pixel(self.n // 2, "x") + self.myt.xside_box / 2.0,
                            self.myt.index_to_pixel(self.m // 2, "y") + self.myt.yside_box / 2.0,
                            self.myt.xside,
                            self.myt.yside)
        self.set_listeners()

    def spacecb(self):
        self.nextGeneration()

    def redisplaycb(self):
        self.nextGeneration(False)

    def savecb(self):
        file_name = 'output_ponyGEGUI_'+str(time.time())
        self.ge.print_to_file(file_name)
        outfile = open(file_name+'.eps','a')
        outfile.write(self.scanvas.postscript())
        outfile.close()
        self.configGUI(NORMAL, NORMAL, NORMAL, NORMAL, DISABLED, DISABLED, "Saved")
    def _destroy(self):
        self.root.destroy()
        sys.exit()

    def configGUI(self, menu, next, save, redisplay, stop, clear, txt="", color="blue"):

        self.next_btn.config(state=next)
        if next==NORMAL:
            self.next_btn.config(bg="#d00")
        else:
            self.next_btn.config(bg="#fca")

        self.save_btn.config(state=save)
        if save==NORMAL:
            self.save_btn.config(bg="#d00")
        else:
            self.save_btn.config(bg="#fca")

        self.redisplay_btn.config(state=redisplay)
        if redisplay==NORMAL:
            self.redisplay_btn.config(bg="#d00")
        else:
            self.redisplay_btn.config(bg="#fca")

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

    def nextGeneration(self, step=True):
        self.refreshCanvas()
        self.dirty = True
        turtle.TurtleScreen._RUNNING = True
        self.screen.clear()
        self.screen.mode("standard")
        self.state = RUNNING

        self.selected = []
        self.configGUI(DISABLED, DISABLED, DISABLED, DISABLED, NORMAL, DISABLED,
                       "Drawing, please wait...", "red")

        if step:
            self.ge.set_fitnesses(self.fitness)
            self.ge.step()

        self.myt = MyTurtle(650, 650, 3, 3)
        turtle.clear()
        for i in range(self.n):
            for j in range(self.m):
                self.setUnselected(i, j)
                #Drawing l-system
#                phenotype = 'angle=6%d\ndepth=%d\nstep_size=10\ncircle_angle=20.5\naxiom=F\nF=F-F++F-F'%((i*j),(i*j))
                phenotype = self.ge.individuals[i*self.n+j].phenotype
                valid = self.draw_phenotype(
                    phenotype,
                    self.myt.index_to_pixel(i, "x") + self.myt.xside_box / 2.0,
                    self.myt.index_to_pixel(j, "y") + self.myt.yside_box / 2.0,
                    self.myt.xside,
                    self.myt.yside)
                if not valid:
                    self.setInvalid(i, j)

        self.configGUI(NORMAL, NORMAL, NORMAL, NORMAL, DISABLED, DISABLED, INSTRUCTIONS)
        self.set_listeners()

    def set_listeners(self):
        turtle.onscreenclick(self.clickcb, 1)
        # In OSX, ctrl-click doesn't give a button-3 click. Instead,
        # trackpad users have to enable secondary click in system
        # preferences. Then, right-clicking on trackpad generates a
        # button-2 click. I don't know what one-button mouse users can
        # do?
        turtle.onscreenclick(self.rightclickcb, 2)
        # On Linux, right-click generates a button-3 click.
        turtle.onscreenclick(self.rightclickcb, 3)
        turtle.onkey(self.spacecb, "space")
        turtle.onkey(self.savecb, "s")
        turtle.onkey(self.redisplaycb, "r")
        turtle.listen()

    # Return True if a non-null drawing was produced, ie the
    # individual is valid (phenotype is not None), phenotype does not
    # exceed maximum length, and drawing contains some commands which
    # actually paint something.
    def draw_phenotype(self, phenotype, x, y, w, h):
        if phenotype is None:
            print("Invalid phenotype")
            return False
        print(phenotype)
        phenotype = "drawing." + phenotype
        _drawing = eval(phenotype)
        non_null = _drawing.draw(x, y, w, h, max_length=8000,
                                 force_fields=ATTRACTORS)
        return non_null

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

    def setInvalid(self, i, j):
        self.fitness[i*self.n + j] = -10.0

    def clearCanvas(self):
        self.refreshCanvas()
        self.screen._delete("all")
        self.scanvas.config(cursor="")
        self.configGUI(NORMAL, NORMAL, NORMAL, NORMAL, DISABLED, DISABLED)

    def stopIt(self):
        if self.exitflag:
            self.clearCanvas()
            self.exitflag = False
            self.configGUI(NORMAL, NORMAL, NORMAL, NORMAL, DISABLED, NORMAL,
                           "STOPPED!", "red")
            turtle.TurtleScreen._RUNNING = False
            #print "stopIT: exitflag = True"
        else:
            turtle.TurtleScreen._RUNNING = False
            #print "stopIt: exitflag = False"

if __name__ == '__main__':
    import getopt
    try:
        #FIXME help option
        print(sys.argv)
        OPTS, ARGS = getopt.getopt(sys.argv[1:], "a:G:g:",
                                   ["attractors=", "grammar_type=", "grammar_file="])
    except getopt.GetoptError as err:
        print(str(err))
        #FIXME usage
        sys.exit(2)
    ATTRACTORS = []
    GRAMMAR_TYPE = LSystem
    GRAMMAR_FILE = "grammars/lsystem.bnf"
    for opt, arg in OPTS:
        print(opt, arg)
        if opt in ("-a", "--attractors"):
            # Attractors are read as dictionarys containing type,
            # effect and size, they are separated by :, e.g -a
            # "type=positive, effect=gravity, x=0, y=0,
            # size=10:type=negative, effect=color, x=10, y=20,
            # size=20"
            # TODO very long code for the attractor parsing
            number_values = ('x', 'y', 'size')
            splits = arg.split(':')
            for split in splits:
                if len(split) > 0:
                    attractor = {}
                    items = split.split(',')
                    for item in items:
                        key, value = [item.strip() for item in item.split('=')]
                        if key in number_values:
                            value = float(value)
                        attractor[key] = value
                    ATTRACTORS.append(attractor)
        elif opt in ("-G", "--grammar_type"):
            # Default grammar type is LSystem
            GRAMMAR_TYPE = eval(arg)
        elif opt in ("-g", "--grammar_file"):
            # Default grammar type is lsystem.bnf
            GRAMMAR_FILE = arg
        else:
            assert False, "unhandeled option"
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
