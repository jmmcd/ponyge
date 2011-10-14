class Tree:

    def __init__(self, expr):
        if len(expr) == 1:
            self.root = expr[0]
            self.children = []
        else:
            self.root = expr[0]
            self.children = []
            for child in expr[1:]:
               self.children.append(Tree(child))
                    
            

    
