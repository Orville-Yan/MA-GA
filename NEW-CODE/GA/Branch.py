import torch
import pandas as pd
import numpy as np
from deap import gp, base, tools, creator
from OP import *
from GA_tools import *

class Branch:
    def __init__(self, mp_root: 'MP_Root', population_size: int, OP_func_list: list, terminal_type: type):
        self.input = mp_root
        self.population_size = population_size
        self.OP_func_list = OP_func_list
        self.terminal_type = terminal_type

    def generate_toolbox(self):
        self.pset = gp.PrimitiveSetTyped("MAIN", [TypeB]*len(self.input), TypeD)

        for name in self.OP_func_list:
            func = getattr(OP_B2D, name) 
            self.pset.addPrimitive(func, TypeB, TypeD)

        self.pset.addTerminal(np.random.rand(1)[0], self.terminal_type)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create(self.__class__.__name__, gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=1)  
        self.toolbox.register(self.__class__.__name__, tools.initIterate, creator.__name__, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.__name__)
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
        super().__init__(mp_root, population_size, OP_B2D_func_list, TypeB) 

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
        super().__init__(mp_root, population_size, OP_B2D_func_list, TypeB)

class M_Branch_MPDP2D(Branch):
    def __init__(self, mp_root: 'MP_Root', population_size: int):
        OP_BA2D_func_list = [
            'Mmask_day_plus',
            'Mmask_day_sub'
        ]
        super().__init__(mp_root, population_size, OP_BA2D_func_list, TypeB)

    def generate_toolbox(self):
        self.pset = gp.PrimitiveSetTyped("MAIN", [TypeB, TypeA], TypeD)
        for name in self.OP_BA2D_func_list:
            func = getattr(OP_BA2D, name) 
            self.pset.addPrimitive(func, [TypeB, TypeA], TypeD)
        
        self.pset.addTerminal(np.random.rand(1)[0], TypeA) 
        self.pset.addTerminal(np.random.rand(1)[0], TypeB) 

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("M_Branch_MPDP2D", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=1)  
        self.toolbox.register("M_Branch_MPDP2D", tools.initIterate, creator.M_Branch_MPDP2D, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.M_Branch_MPDP2D)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

    def generate_population(self):
        self.individuals_code = self.toolbox.population(n=self.population_size)
        self.individuals_code, self.individuals_str = change_name(self.individuals_code, self.input)
class M_Branch_MVDV2D(Branch):
    def __init__(self, mp_root: 'MP_Root', population_size: int):
        OP_BA2D_func_list = [
            'Mmask_day_plus',
            'Mmask_day_sub'
        ]
        super().__init__(mp_root, population_size, OP_BA2D_func_list, TypeB)

    def generate_toolbox(self):
        self.pset = gp.PrimitiveSetTyped("MAIN", [TypeB, TypeA], TypeD)
        for name in self.OP_BA2D_func_list:
            func = getattr(OP_BA2D, name) 
            self.pset.addPrimitive(func, [TypeB, TypeA], TypeD)
        
        self.pset.addTerminal(np.random.rand(1)[0], TypeA) 
        self.pset.addTerminal(np.random.rand(1)[0], TypeB) 

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("M_Branch_MVDV2D", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=1)  
        self.toolbox.register("M_Branch_MVDV2D", tools.initIterate, creator.M_Branch_MVDV2D, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.M_Branch_MVDV2D)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

    def generate_population(self):
        self.individuals_code = self.toolbox.population(n=self.population_size)
        self.individuals_code, self.individuals_str = change_name(self.individuals_code, self.input)
