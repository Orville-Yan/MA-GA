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

