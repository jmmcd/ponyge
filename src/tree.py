import random, copy
import ponyge as p
arities = {"*":2, "+":2, "-": 2, "1":0, "2":0, "3":0, "4":0, "5":0, "6":0,
           "7":0, "8":0, "9":0, "0":0, "sin":1, "cos":1, "tan":1}

class Tree:

    def __init__(self, expr, parent, maxDepth = 10):
        self.parent = parent
        self.maxDepth = maxDepth
        if len(expr) == 1:
            self.root = expr[0]
            self.children = []
        else:
            self.root = expr[0]
            self.children = []
            for child in expr[1:]:
                if type(child) == tuple:
                    self.children.append(Tree(child, self))
                else:
                    self.children.append(Tree((child,), self))
        self.grammar = None
    def __str__(self):
        result = "("
        result += str(self.root)
        for child in self.children:
            if len(child.children) > 0:
                result += " " + str(child)
            else:
                result += " " + str(child.root)
        result += ")"
        return result
    def getDepth(self):
        count = 0
        currentParent = self.parent
        while currentParent != None:
            count += 1
            currentParent = currentParent.parent
        return count
    def getRandom(self, prob):
        r = random.random()
        if r < prob or len(self.children) == 0:
            return self
        else:
            return random.choice(self.children).getRandom(prob)

    def grammarDerivation(self, grammar):
        self.grammar = grammar
        productions = grammar.rules[self.root]
        chosen_prod = random.choice(productions)
        self.children = []

        #checks max depth
        if self.getDepth() == self.maxDepth:
            while all([sym[1] == grammar.T for sym in chosen_prod]) == False:
                chosen_prod = random.choice(productions)
            
        for i in range(len(chosen_prod)):
            symbol = chosen_prod[i]
            if symbol[1] == grammar.T: #if the right hand side is a terminal
                self.children.append(Tree((symbol[0],),self))
            elif symbol[1] == grammar.NT: # if the right hand side is a non-terminal
                self.children.append(Tree((symbol[0],),self))
                self.children[-1].grammarDerivation(grammar)

    def getOutput(self):
        output = []
        for child in self.children:
            if child.children == []:
                output.append(child.root)
            else:
                output += child.getOutput()
        return "".join(output)
            
    def grow(self):
        if self.getDepth() == self.maxDepth: 
            while arities[self.root] != 0:
                self.root = random.choice[arities.keys()]
            return self
        
        label = random.choice(arities.keys())
        arity = arities[label]
        self.root = label
        self.children = []
        for c in range(arity):
            self.children.append(Tree(("1",), self))
            self.children[c].grow()
        return self
    
    def mutate(self):
        #choose random node
        tree = self.getRandom(0.2)
        
        #grow from random node
        tree.grow()

    def grammarMutate(self):
        #choose random node
        tree = self.getRandom(0.2)
        while tree.root in self.grammar.terminals:
            tree = self.getRandom(0.2)
        
        #grow(based on grammar) from random node
        tree.grammarDerivation(self.grammar)

    def getLabels(self):
        labels = [self.root]

        for c in self.children:
            labels.extend(c.getLabels())
        return set(labels)

def crossover(tree1, tree2):
        tree1 = copy.deepcopy(tree1)
        tree2 = copy.deepcopy(tree2)
        
        labels1 = tree1.getLabels()
        labels2 = tree2.getLabels()
        intersection = labels1.intersection(labels2)

        intersection = filter(lambda x: x in tree1.grammar.non_terminals, intersection)
        print intersection

        #see if there is a label to do a crossover
        if len(intersection) != 0:
            t1 = tree1.getRandom(0.2)

            #finds the nodes to crossover at

            while t1.root not in intersection:
                t1 = tree1.getRandom(0.2)

            t2 = tree2.getRandom(0.2)
            while t2.root != t1.root:
                t2 = tree2.getRandom(0.2)
                
            p1 = t1.parent
            p2 = t2.parent
            
            # when the crossover is between the entire tree of both tree1 and tree2
            if p1 == None and p2 == None:
                return [t2, t1]
            #when only t1 is the entire tree1
            elif p1 == None:
                tree1 = t2
                i2 = p2.children.index(t2)
                p2.children[i2] = t1
                t1.parent = p2
            #when only t2 is the entire tree2
            elif p2 == None:
                tree2 = t1
                i1 = p1.children.index(t1)
                p1.children[i1] = t2
                t2.parent = p1
            #when the crossover node for both trees is not the entire tree
            else:
                i1 = p1.children.index(t1)
                i2 = p2.children.index(t2)

                p1.children[i1] = t2
                p2.children[i2] = t1

                t2.parent = p1
                t1.parent = p2

        return [tree1, tree2]

def getDerivation(grammar):
    tree = Tree((str(grammar.start_rule[0]),), None)
    tree.grammarDerivation(grammar)
    return (tree, tree.getOutput())

def crossoverTests():
     tree1 = Tree(("+",("*", "2", "3",),"5"), None)
     tree2 = Tree(("-","7",("+","1","6")), None)

     #1 common label
     crossList = crossover(tree1, tree2)
     assert str(crossList[0]) == "(+ 1 6)"
     assert str(crossList[1]) == "(- 7 (+ (* 2 3) 5))"

     tree3 = Tree(("-", "7", "1"), None)

     #no common label
     crossList = crossover(tree1, tree3)
     assert str(crossList[0]) == str(tree1)
     assert str(crossList[1]) == str(tree3)

     tree4 = Tree(("-","7",("+","1",("+", "4", "8"))), None)

     #1 common label multiple spots in tree
     crossList = crossover(tree1, tree4)
     assert str(crossList[0]) != str(tree1)
     assert str(crossList[1]) != str(tree4)

     tree5 = Tree(("-","7",("+","1",("+", "3", "5"))), None)

     #multiple common labels
     crossList = crossover(tree1, tree5)
     assert str(crossList[0]) != str(tree1)
     assert str(crossList[1]) != str(tree5)

     tree6 = Tree(("-", "3", "4"), None)

     #common label is a number(arity = 0) ==> do nothing
     crossList = crossover(tree1, tree6)
     assert str(crossList[0]) == str(tree1)
     assert str(crossList[1]) == str(tree6)

def grammarDerivationTests():
    grammar1 = p.Grammar("grammars/boolean.pybnf")
    tree,output = getDerivation(grammar1)
    print tree
    print output
    
         
if __name__ == '__main__':
    #crossoverTests()
    grammarDerivationTests()
