import random
arities = {"*":2, "+":2, "-": 2, "1":0, "2":0, "3":0, "4":0, "5":0, "6":0,
           "7":0, "8":0, "9":0, "0":0, "sin":1, "cos":1, "tan":1}
arities = {"*":4, "+":4, "-": 4, "1":0, "sin":1, "cos":1, "tan":1}

class Tree:

    def __init__(self, expr, parent, maxDepth = 3):
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
    def __str__(self):
        result = "("
        result += str(self.root)
        for child in self.children:
            if len(child.children) > 0:
                result += " " + child.__str__()
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
            
    def grow(self):
        print self.getDepth()
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

    def getLabels(self):
        labels = [self.root]

        for c in self.children:
            labels.extend(c.getLabels())
        return set(labels)

def crossover(tree1, tree2):
        # equal labels not arities??????
        labels1 = tree1.getLabels()
        labels2 = tree2.getLabels()
        intersection = labels1.intersection(labels2)

        if len(intersection) != 0:
            t1 = tree1.getRandom(0.2)
            t2 = tree2.getRandom(0.2)

            while t1.root != t2.root:
                t1 = tree1.getRandom(0.2)
                t2 = tree2.getRandom(0.2)

            print t1
            print t2

            temp1 = t1
            temp2 = t2
            t1 = temp2
            t2 = temp1

        return [tree1, tree2]

