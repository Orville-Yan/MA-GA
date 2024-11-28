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


class Correction:
    def __init__(self, deap_formula_str_list, pset):
        self.deap_formula_str_list = deap_formula_str_list
        self.deap_formula_code_list = [gp.PrimitiveTree.from_string(k, pset) for k in deap_formula_str_list]
        self.pset = pset

    def trancate_formula(self, tuple_list, dict, replace_terminal):
        y = []
        i = 0
        k = 0
        for s in tuple_list:
            if isinstance(s[0], gp.Primitive):
                k += 1
        if k > 1:
            y.append(tuple_list[i])
            i += 1

        while i < len(tuple_list):
            if (i + 1 < len(tuple_list)):
                if isinstance(tuple_list[i][0], gp.Primitive) & isinstance(tuple_list[i + 1][0], gp.Terminal):
                    t = Tree(tuple_list[i][1])
                    for j in range(i, i + tuple_list[i][0].arity + 1):
                        t.get_leaf(dict[tuple_list[j][1]])

                    for j in range(i + 1, i + tuple_list[i][0].arity + 1):
                        t.get_node(dict[tuple_list[j][1]])

                    dict[tuple_list[i][1]] = t
                    y.append((replace_terminal, tuple_list[i][1]))
                    i += tuple_list[i][0].arity + 1
                else:
                    y.append(tuple_list[i])
                    i += 1
            else:
                y.append(tuple_list[i])
                i += 1
        return y, dict

    def test_no_prim(self, tuple_list):
        num_count = 0
        for item in tuple_list:
            if isinstance(item[0], gp.Primitive):
                num_count += 1
        return num_count

    def transform_formala2tree_constructure(self,deap_formula_code):
        x = [(deap_formula_code[i], i) for i in range(len(deap_formula_code))]
        area = {i: i for i in range(len(deap_formula_code))}
        while self.test_no_prim(x) > 0:
            x, area = self.trancate_formula(x, area, self.byword_2)
        return area
