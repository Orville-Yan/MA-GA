from GA_tools import *
from OP.ToA import *
from OP.Others import *
import numpy as np
from deap import gp, base, tools, creator
class Subtree:
    def __init__(self,population_size=10):
        self.OP_BD2A_func_list = ['D_Minute_area_mean', 'D_Minute_area_std', 'D_Minute_area_sum', 'D_Minute_area_prod']
        self.OP_B2A_func_list = ['D_Minute_std', 'D_Minute_mean', 'D_Minute_trend']
        self.OP_BBD2A_func_list = ['D_Minute_area_weight_mean', 'D_Minute_area_corr', 'D_Minute_area_rankcorr','D_Minute_area_bifurcate_mean', 'D_Minute_area_bifurcate_std']
        self.OP_BB2A_func_list = ['D_Minute_corr', 'D_Minute_weight_mean']
        self.population_size = population_size

    def generate_toolbox_with_mask(self):
        self.pset_with_mask = gp.PrimitiveSetTyped("MAIN_with_mask", [TypeB, TypeD], TypeA)

        for func_name in self.OP_BD2A_func_list:
            func = getattr(OP_BD2A, func_name)
            self.pset_with_mask.addPrimitive(func, [TypeB, TypeD], TypeA, name=func_name)

        for func_name in self.OP_BBD2A_func_list:
            func = getattr(OP_BBD2A, func_name)
            self.pset_with_mask.addPrimitive(func, [TypeB, TypeB, TypeD], TypeA, name=func_name)

        self.pset_with_mask.addTerminal(np.random.rand(1)[0], TypeA)

        creator.create("FitnessMax_with_mask", base.Fitness, weights=(1.0,))
        creator.create("Subtree_withMask", gp.PrimitiveTree, fitness=creator.FitnessMax_with_mask, pset=self.pset_with_mask)

        self.toolbox_with_mask = base.Toolbox()
        self.toolbox_with_mask.register("expr", gp.genHalfAndHalf, pset=self.pset_with_mask, min_=1, max_=1)
        self.toolbox_with_mask.register("individual", tools.initIterate, creator.Subtree_withMask, self.toolbox_with_mask.expr)
        self.toolbox_with_mask.register("population", tools.initRepeat, list, self.toolbox_with_mask.individual)
        self.toolbox_with_mask.register("compile", gp.compile, pset=self.pset_with_mask)

    def generate_toolbox_no_mask(self):
        self.pset_no_mask = gp.PrimitiveSetTyped("MAIN_no_mask", [TypeB], TypeA)

        for func_name in self.OP_B2A_func_list:
            func = getattr(OP_B2A, func_name)
            self.pset_no_mask.addPrimitive(func, [TypeB], TypeA, name=func_name)

        for func_name in self.OP_BB2A_func_list:
            func = getattr(OP_BB2A, func_name)
            self.pset_no_mask.addPrimitive(func, [TypeB, TypeB], TypeA, name=func_name)

        self.pset_no_mask.addTerminal(np.random.rand(1)[0], TypeA)

        creator.create("FitnessMax_no_mask", base.Fitness, weights=(1.0,))
        creator.create("Subtree_noMask", gp.PrimitiveTree, fitness=creator.FitnessMax_no_mask, pset=self.pset_no_mask)

        self.toolbox_no_mask = base.Toolbox()
        self.toolbox_no_mask.register("expr", gp.genHalfAndHalf, pset=self.pset_no_mask, min_=1, max_=1)
        self.toolbox_no_mask.register("individual", tools.initIterate, creator.Subtree_noMask, self.toolbox_no_mask.expr)
        self.toolbox_no_mask.register("population", tools.initRepeat, list, self.toolbox_no_mask.individual)
        self.toolbox_no_mask.register("compile", gp.compile, pset=self.pset_no_mask)

    def generate_population_with_mask(self):
        self.individuals_code_with_mask = self.toolbox_with_mask.population(n=self.population_size)
        self.individuals_code_with_mask, self.individuals_str_with_mask = change_name(self.individuals_code_with_mask, self.input)

    def generate_population_no_mask(self):
        self.individuals_code_no_mask = self.toolbox_no_mask.population(n=self.population_size)
        self.individuals_code_no_mask, self.individuals_str_no_mask = change_name(self.individuals_code_no_mask, self.input)
