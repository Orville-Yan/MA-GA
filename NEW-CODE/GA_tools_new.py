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

class TypeF(torch.Tensor):
    def __new__(cls,*args,**kwargs):
        return super(TypeF,cls).__new__(cls,*args,dtype=torch.int,**kwargs)
    pass

class TypeG(torch.Tensor):
    def __new__(cls,*args,**kwargs):
        return super(TypeF,cls).__new__(cls,*args,dtype=torch.float32,**kwargs)
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
    def __init__(self, deap_formula_str_list, pset) -> None:
        self.deap_formula_str_list = deap_formula_str_list
        self.deap_formula_code_list = [gp.PrimitiveTree.from_string(k, pset) for k in deap_formula_str_list]
        self.pset = pset
        self.Tree_Structure = None
        self.seed = {}
        self.root = {}
        self.Tree = {}
        
    def get_Tree_Structure(self):
        self.Tree_Structure = {}
        self.seed = {}
        self.root = {}
        self.Tree = {}

        for i, tree in enumerate(self.deap_formula_code_list):
            tree_structure, _ = self.split_tree(tree)
            self.Tree_Structure[i] = tree_structure
            self.seed[i] = str(tree)
            self.root[i] = tree_structure[0].name
            self.Tree[i] = self.build_tree(tree_structure)

    def split_tree(self, tree):
        if isinstance(tree, gp.PrimitiveTree):
            root = tree[0]
            children = []
            start_index = 1
            for _ in range(root.arity):
                child_tree, end_index = self.split_tree(tree[start_index:])
                children.append(child_tree)
                start_index = end_index
            return (root, children), start_index
        else:
            return tree, 1

    def build_tree(self, tree_structure):
        if isinstance(tree_structure, tuple):
            root, children = tree_structure
            subtree = [root]
            for child in children:
                subtree.extend(self.build_tree(child))
            return subtree
        else:
            return [tree_structure]
            
deap_formula_str_list = [
    "at_div(M_ts_mean_left_neighbor(M_O, 5, -1), M_ts_mean_right_neighbor(M_C, 10, 1))",
    "at_div(M_ts_mean_right_neighbor(M_C, 10, 1), M_ts_mean_left_neighbor(M_O, 5, -1))"
]

pset = gp.PrimitiveSetTyped("MAIN", [TypeB] * 2, TypeB)
pset.addPrimitive(lambda x, y: x / y, [TypeB, TypeB], TypeB, name="at_div")
pset.addPrimitive(lambda x, y, z: f"M_ts_mean_left_neighbor({x}, {y}, {z})", [TypeB, int, int], TypeB, name="M_ts_mean_left_neighbor")
pset.addPrimitive(lambda x, y, z: f"M_ts_mean_right_neighbor({x}, {y}, {z})", [TypeB, int, int], TypeB, name="M_ts_mean_right_neighbor")
pset.addTerminal("M_O", TypeB)
pset.addTerminal("M_C", TypeB)

tree = Tree(deap_formula_str_list, pset)
tree.get_Tree_Structure()

print("Seed:", tree.seed)
print("Root:", tree.root)
print("Tree:", tree.Tree)
print("Tree_Structure:", tree.Tree_Structure)

# class Correction:
#     def __init__(self, deap_formula_str_list, pset):
#         self.deap_formula_str_list = deap_formula_str_list
#         self.deap_formula_code_list = [gp.PrimitiveTree.from_string(k, pset) for k in deap_formula_str_list]
#         self.pset = pset

#     def trancate_formula(self, tuple_list, dict, replace_terminal):
#         y = []
#         i = 0
#         k = 0
#         for s in tuple_list:
#             if isinstance(s[0], gp.Primitive):
#                 k += 1
#         if k > 1:
#             y.append(tuple_list[i])
#             i += 1

#         while i < len(tuple_list):
#             if (i + 1 < len(tuple_list)):
#                 if isinstance(tuple_list[i][0], gp.Primitive) & isinstance(tuple_list[i + 1][0], gp.Terminal):
#                     t = Tree(tuple_list[i][1])
#                     for j in range(i, i + tuple_list[i][0].arity + 1):
#                         t.get_leaf(dict[tuple_list[j][1]])

#                     for j in range(i + 1, i + tuple_list[i][0].arity + 1):
#                         t.get_node(dict[tuple_list[j][1]])

#                     dict[tuple_list[i][1]] = t
#                     y.append((replace_terminal, tuple_list[i][1]))
#                     i += tuple_list[i][0].arity + 1
#                 else:
#                     y.append(tuple_list[i])
#                     i += 1
#             else:
#                 y.append(tuple_list[i])
#                 i += 1
#         return y, dict

#     def test_no_prim(self, tuple_list):
#         num_count = 0
#         for item in tuple_list:
#             if isinstance(item[0], gp.Primitive):
#                 num_count += 1
#         return num_count

#     def transform_formala2tree_constructure(self,deap_formula_code):
#         x = [(deap_formula_code[i], i) for i in range(len(deap_formula_code))]
#         area = {i: i for i in range(len(deap_formula_code))}
#         while self.test_no_prim(x) > 0:
#             x, area = self.trancate_formula(x, area, self.byword_2)
#         return area
