import torch
from typing import Union
from deap import gp, creator, base, tools

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

class TypeG(torch.Tensor):
    def __new__(cls,*args,**kwargs):
        return super(TypeG,cls).__new__(cls,*args,dtype=torch.float32,**kwargs)

    
class chaotic_map:
    def __init__(self):
        self.map_list = ["chebyshev_map","circle_map","iterative_map","logistic_map","piecewise_map","sine_map","singer_map",
    "tent_map","spm_map","tent_logistic_cosine_map","sine_tent_cosine_map","logistic_sine_cosine_map",
    "cubic_map","logistic_tent_map","bernoulli_map"]

    @staticmethod
    def chebyshev_map(x, a=4):#(-1,1) 过于接近0/1则不会更新
        return torch.cos(a * torch.acos(x))

    @staticmethod
    def circle_map(x, a=0.5, b=2.2):#(0,1)
        return (x + a - (b / 2 / torch.pi * torch.sin(2 * torch.pi * x))) % 1

    @staticmethod
    def iterative_map(x, a=0.7):#(-1,1)
        return torch.sin(a * torch.pi / x)

    @staticmethod
    def logistic_map(x, a=4):#(0,1)
        return a * x * (1 - x)

    @staticmethod
    def piecewise_map(x, d=0.3):#(0,1) 过于接近0/1则不会更新
        x = torch.where(x > 0.5, 1 - x, x)
        return torch.where(x < d, x / d,(x - d) / (0.5 - d))

    @staticmethod
    def sine_map(x, a=4):#(0,1) 可能数值溢出 过于接近0/1则不会更新
        return a / 4 * torch.sin(torch.pi * x)

    @staticmethod
    def singer_map(x, a=1.07):#(0,1) 过于接近1会数值溢出
        return a * (7.86 * x - 23.31 * x**2 + 28.75 * x**3 - 13.302875 * x**4)

    @staticmethod
    def tent_map(x, a=0.4):#(0,1)
        return torch.where(x < a, x / a, (1 - x) / (1 - a))

    @staticmethod
    def spm_map(x, eta=0.4, mu=0.3, r=torch.rand(1)):#(0,1)
        x = torch.where(x > 0.5, 1 - x, x)
        return torch.where(x < eta,(x / eta + mu * torch.sin(torch.pi * x) + r) % 1,
        (x / eta / (0.5 - eta) + mu * torch.sin(torch.pi * x) + r) % 1)

    @staticmethod
    def tent_logistic_cosine_map(x, r=0.7):#(0,1) 可能数值溢出 
        return torch.where(x<0.5,torch.cos(torch.pi * (2 * r * x + 4 * (1 - r) * x * (1 - x) - 0.5)),
        torch.cos(torch.pi * (2 * r * (1 - x) + 4 * (1 - r) * x * (1 - x) - 0.5)))

    @staticmethod
    def sine_tent_cosine_map(x, r=0.7):#(0,1) 可能数值溢出
        return torch.where(x < 0.5,torch.cos(torch.pi * (r * torch.sin(torch.pi * x) + 2 * (1 - r) * x - 0.5)),
        torch.cos(torch.pi * (r * torch.sin(torch.pi * x) + 2 * (1 - r) * (1 - x) - 0.5)))

    @staticmethod
    def logistic_sine_cosine_map(x, r=0.7):#(0,1) 可能数值溢出
        return torch.cos(torch.pi * (4 * r * x * (1-x) + (1-r) * torch.sin(torch.pi * x) - 0.5))

    @staticmethod
    def cubic_map(x, a=2.595):#(0,1) 过于接近0/1可能不会更新
        return a * x * (1 - x**2)

    @staticmethod
    def logistic_tent_map(x, r=0.3):#(0,1) 过于接近0/1可能不会更新
        return torch.where(x < 0.5,(r * x * (1 - x) + (4 - r) * x / 2) % 1,(r * x * (1 - x) + (4 - r) * (1 - x) / 2) % 1)

    @staticmethod
    def bernoulli_map(x, a=0.4):#(0,1) 过于接近1则不会更新
        return torch.where(x <= 1 - a, x / (1 - a), (x - 1 + a) / a)
        
    
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

class Leaf:
    pass

class Acyclic_Tree:
    def __init__(self,deap_str,pset):
        self.tree=deap_str
        self.root_node=None #gp.Primitives
        self.nodes=[] #[Union[Leaf,Acyclic_Tree]]

        self.abbreviation=None

# class Tree:
#     def __init__(self, root):
        
#         if not isinstance(root,gp.Primitive):
#             raise CustomError('Build Tree without a root!')
#         self.root = root
#         self.name = root.name
#         self.node = []
#         self.node_name = []
#         self.roots = ""
#         self.seed = []

#     def get_node(self, node):
#         if isinstance(node, Tree):
#             self.node.append(node)
#             self.node_name.append(node.root.name)
#         else:
#             self.node.append(node)
#             self.node_name.append(node.name)
            
#     def get_root(self):
#         if len(self.node) == 0:
#             raise CustomError(f'Something went wrong when building the tree with root {self.name}')
#         output = f"{self.name}"
#         output += '('
#         for i, node in enumerate(self.node):
#             if isinstance(node,Tree):
#                 output += node.get_root()
#             else:
#                 output += node.name
#             if i < len(self.node) - 1:
#                 output += ', '
#             # else:
#         output +=')'
#         self.roots = output
#         return output
    
#     def get_seed(self):
#         if len(self.node) == 0:
#             raise CustomError(f'Something went wrong when building the tree with root {self.name}')
#         self.seed = []
#         flag = True if self.root.name in seed_name else False
#         for i, node in enumerate(self.node):
#             if isinstance(node,Tree):
#                 # if node.root.name in seed_name:
#                 seed = node.get_seed()
#                 self.seed.extend(seed)
#                 # else:
#                 #     flag = False
#                 #     seed = node.get_seed()
#                 #     self.seed.extend(seed)
#             else:
#                 self.seed.append(node.name)
#         if flag:
#             self.seed.append(self.get_root())
#         return self.seed
            
    
# class Treebuilder: 
#     def __init__(self,deap_formula_str_list,pset) -> None:
#         self.deap_formula_str_list = deap_formula_str_list
#         self.deap_formula_code_list = [gp.PrimitiveTree.from_string(k, pset) for k in deap_formula_str_list]
#         self.feature_names = [[node.name for node in self.deap_formula_code_list[k]] for k in range(len(deap_formula_str_list))]
#         self.pset = pset
#         pass
           
#     def get_tree(self,x):
#         root_cache = []
#         terminals = [0]
#         root_terminals = []
#         output = ''
#         for i, node in enumerate(self.deap_formula_code_list[x]):
#             if i == 0 & isinstance(node, gp.Primitive):
#                 output += node.name + '('
#                 ode = Tree(node)
#                 root_terminals.append(ode)
#                 root_cache.append(ode)
#                 terminals.append(node.arity)
#                 continue
#             if isinstance(node, gp.Primitive):
#                 terminals.append(node.arity)
#                 ode = Tree(node)
#                 root_terminals[-1].get_node(ode)
#                 root_terminals.append(ode)
#                 root_cache.append(ode)
#                 output += node.name + '('
#             else:
#                 if isinstance(node, gp.Terminal):
#                     output += self.feature_names[x][i]
                    
#                 elif isinstance(node, int):
#                     if self.feature_names is None:
#                         output += self.feature_names[x][i]
#                 else:
#                     output += self.feature_names[x][i]
#                 root_terminals[-1].get_node(node)
#                 terminals[-1] -= 1
#                 while terminals[-1] == 0:
#                     terminals.pop()
#                     terminals[-1] -= 1
#                     root_terminals.pop()
#                     output += ')'
#                 if i != len(self.deap_formula_code_list[x]) - 1:
#                     output += ', '
#         self.output = output
#         self.root_cache = root_cache
