import torch
import pandas as pd
import numpy as np
from deap import gp, base, tools, creator
from OP import *
from GA_tools import *

class Branch:
    def __init__(self, mp_root: 'MP_Root', population_size: int,OP_func,OP_func_list:list,OP_input_type,OP_output_type):
        self.input = mp_root
        self.population_size = population_size
        self.OP_func = OP_func
        self.OP_func_list = OP_func_list
        self.OP_input_type = OP_input_type
        self.OP_output_type = OP_output_type
        self.pset = None
        self.toolbox = None
        self.int_values = [torch.tensor(i,dtype=torch.int) for i in [2, 3, 5, 8, 10 ,30]]   

    def generate_toolbox(self):
        class_name = self.__class__.__name__ 
        self.pset = gp.PrimitiveSetTyped(class_name, self.OP_input_type*len(self.input), self.OP_output_type)
        for name in self.OP_func_list:
            func = getattr(self.OP_func,name)
            self.pset.addPrimitive(func, self.OP_input_type, self.OP_output_type)
        for input_type in  set(self.OP_input_type):
            for constant in self.int_values:
                self.pset.addTerminal(constant, input_type) 
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create(class_name, gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=1)  
        self.toolbox.register(class_name, tools.initIterate, creator.class_name, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.class__name)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

    def generate_population(self):
        self.individuals_code = self.toolbox.population(n=self.population_size)
        self.individuals_code, self.individuals_str = change_name(self.individuals_code, self.input)


class M_Branch_MP2D(Branch):
    def __init__(self, mp_root: 'MP_Root', population_size: int):
        OP_B2D_func_list = [
            'Mmask_min',
            'Mmask_max',
            'Mmask_middle',
            'Mmask_min_to_max',
            'Mmask_mean_plus_std',
            'Mmask_mean_sub_std',
            'Mmask_1h_after_open',
            'Mmask_1h_before_close',
            'Mmask_2h_middle',
            'Mmask_morning',
            'Mmask_afternoon',
        ]
        super().__init__(mp_root, population_size, OP_B2D, OP_B2D_func_list, TypeB,TypeD)


class M_Branch_MV2D(Branch):
    def __init__(self, mp_root: 'MP_Root', population_size: int):
        OP_B2D_func_list = [
            'Mmask_min',
            'Mmask_max',
            'Mmask_middle',
            'Mmask_min_to_max',
            'Mmask_mean_plus_std',
            'Mmask_mean_sub_std',
            'Mmask_1h_after_open',
            'Mmask_1h_before_close',
            'Mmask_2h_middle',
            'Mmask_morning',
            'Mmask_afternoon',
        ]
        super().__init__(mp_root, population_size, OP_B2D, OP_B2D_func_list, TypeB,TypeD)


class M_Branch_MPDP2D(Branch):
    def __init__(self, mp_root: 'MP_Root', population_size: int):
        OP_BA2D_func_list = [
            'Mmask_day_plus',
            'Mmask_day_sub'
        ]
        super().__init__(mp_root, population_size, OP_BA2D,OP_BA2D_func_list,[TypeB,TypeA],TypeD)




class M_Branch_MVDV2D(Branch):
    def __init__(self, mp_root: 'MP_Root', population_size: int):
        OP_BA2D_func_list = [
            'Mmask_day_plus',
            'Mmask_day_sub'
        ]
        super().__init__(mp_root, population_size, OP_BA2D,OP_BA2D_func_list,[TypeB,TypeA],TypeD)
