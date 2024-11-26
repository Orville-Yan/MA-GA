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
    
def change_name(formula_list,substitute_list):
    renamed_individual_code = []
    renamed_individual_str = []
    for individual in formula_list:
        for i,node in enumerate(individual):
            if isinstance(node,gp.Terminal) and node.name.startswith('ARG'):
                arg_index = int(node.name[3:])
                new_name = substitute_list[arg_index]
                individual[i]=gp.Terminal(new_name, node.ret, node.value)

        renamed_individual_code.append(individual)
        renamed_individual_str.append(str(individual))
    return renamed_individual_code,renamed_individual_str
    
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

