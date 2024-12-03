import torch
from deap import gp,base,creator,tools

seed_name = [
    'M_ts_mean_right_neighbor','M_O','M_C','M_ts_mean_left_neighbor'
]
class CustomError(Exception):
    def __init__(self, message="发生了自定义错误"):
        self.message = message
        super().__init__(self.message)
        
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
    def __init__(self, root):
        
        if not isinstance(root,gp.Primitive):
            raise CustomError('Build Tree without a root!')
        self.root = root
        self.name = root.name
        self.node = []
        self.node_name = []
        self.roots = ""
        self.seed = []

    def get_node(self, node):
        if isinstance(node, Tree):
            self.node.append(node)
            self.node_name.append(node.root.name)
        else:
            self.node.append(node)
            self.node_name.append(node.name)
            
    def get_root(self):
        if len(self.node) == 0:
            raise CustomError(f'Something went wrong when building the tree with root {self.name}')
        output = f"{self.name}"
        output += '('
        for i, node in enumerate(self.node):
            if isinstance(node,Tree):
                output += node.get_root()
            else:
                output += node.name
            if i < len(self.node) - 1:
                output += ', '
            # else:
        output +=')'
        self.roots = output
        return output
    
    def get_seed(self):
        if len(self.node) == 0:
            raise CustomError(f'Something went wrong when building the tree with root {self.name}')
        self.seed = []
        flag = True if self.root.name in seed_name else False
        for i, node in enumerate(self.node):
            if isinstance(node,Tree):
                # if node.root.name in seed_name:
                seed = node.get_seed()
                self.seed.extend(seed)
                # else:
                #     flag = False
                #     seed = node.get_seed()
                #     self.seed.extend(seed)
            else:
                self.seed.append(node.name)
        if flag:
            self.seed.append(self.get_root())
        return self.seed
            
    
class Treebuilder: 
    def __init__(self,deap_formula_str_list,pset) -> None:
        self.deap_formula_str_list = deap_formula_str_list
        self.deap_formula_code_list = [gp.PrimitiveTree.from_string(k, pset) for k in deap_formula_str_list]
        self.feature_names = [[node.name for node in self.deap_formula_code_list[k]] for k in range(len(deap_formula_str_list))]
        self.pset = pset
        pass
           
    def get_tree(self,x):
        root_cache = []
        terminals = [0]
        root_terminals = []
        output = ''
        for i, node in enumerate(self.deap_formula_code_list[x]):
            if i == 0 & isinstance(node, gp.Primitive):
                output += node.name + '('
                ode = Tree(node)
                root_terminals.append(ode)
                root_cache.append(ode)
                terminals.append(node.arity)
                continue
            if isinstance(node, gp.Primitive):
                terminals.append(node.arity)
                ode = Tree(node)
                root_terminals[-1].get_node(ode)
                root_terminals.append(ode)
                root_cache.append(ode)
                output += node.name + '('
            else:
                if isinstance(node, gp.Terminal):
                    output += self.feature_names[x][i]
                    
                elif isinstance(node, int):
                    if self.feature_names is None:
                        output += self.feature_names[x][i]
                else:
                    output += self.feature_names[x][i]
                root_terminals[-1].get_node(node)
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
                    root_terminals.pop()
                    output += ')'
                if i != len(self.deap_formula_code_list[x]) - 1:
                    output += ', '
        self.output = output
        self.root_cache = root_cache
            
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

builder = Treebuilder(deap_formula_str_list,pset)
builder.get_tree(0)
# print(output)
print('Seed:',builder.root_cache[1].get_seed())
print('Root:',builder.root_cache[0].get_root())

