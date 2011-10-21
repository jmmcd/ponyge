class Tree:

    def __init__(self, expr):
        if len(expr) == 1:
            self.root = expr[0]
            self.children = []
        else:
            self.root = expr[0]
            self.children = []
            for child in expr[1:]:
                if type(child) == tuple:
                    self.children.append(Tree(child))
                else:
                    self.children.append(Tree((child,)))
    def __str__(self):
        result = "("
        result += str(self.root)
        for child in self.children:
            if len(child.children) > 0:
                result += child.__str__()
            else:
                result += " " + str(child.root)
        result += ")"
        return result
