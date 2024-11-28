import torch
from deap import gp,base,creator,tools

class TypeA(torch.Tensor):
    pass
class TypeB(torch.Tensor):
    pass

class TypeC(torch.Tensor):
    pass
class TypeD(torch.Tensor):
    pass

class TypeE(torch.Tensor):
    pass
class TypeF(torch.int):
    pass

class TypeG(torch.float32):
    pass

class Tree:
    def __init__(self, root):
        self.root = root
        self.leaf = []
        self.node = []

    def get_leaf(self, leaf):
        if isinstance(leaf, Tree):
            self.leaf += (leaf.leaf)
        else:
            self.leaf.append(leaf)

    def get_node(self, node):
        if isinstance(node, Tree):
            self.node.append(node.root)
        else:
            self.node.append(node)

