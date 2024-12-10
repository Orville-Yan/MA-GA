import sys
sys.path.append('..')

from Tools.GA_tools import *
from OP import *
import torch


class MP_Root:
    def __init__(self, MP_Seed: [str], population_size=10):
        self.input = MP_Seed
        self.population_size = population_size
        self.OP_B2B_func_list = ['M_cs_rank', 'M_cs_scale', 'M_cs_zscore', 'M_ts_pctchg']
        self.OP_BB2B_func_list = ['M_at_div']
    def generate_toolbox(self):
        self.pset= gp.PrimitiveSetTyped("MAIN", [TypeB] * len(self.input), TypeB)

        for func_name in self.OP_B2B_func_list:
            func = getattr(OP.OP_B2B, func_name, None)
            self.pset.addPrimitive(func, TypeB, TypeB, name=func_name)

        for func_name in self.OP_BB2B_func_list:
            func = getattr(OP.OP_BB2B, func_name, None)
            self.pset.addPrimitive(func, [TypeB, TypeB], TypeB, name=func_name)


        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("MP_Root", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=1)  # 树的深度按需求改
        self.toolbox.register("MP_Root", tools.initIterate, creator.MP_Root, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.MP_Root)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

    def generate_MP_Root(self):
        self.individuals_code = self.toolbox.population(n=self.population_size)
        self.individuals_code, self.individuals_str = change_name(self.individuals_code, self.input)

class MV_Root:
    def __init__(self, MV_Seed: [str], population_size=10):
        self.input = MV_Seed
        self.population_size = population_size
        self.OP_B2B_func_list = ['M_cs_rank', 'M_cs_scale', 'M_cs_zscore', 'M_ts_pctchg']
        self.OP_BB2B_func_list = ['M_at_div']

    def generate_toolbox(self):
        self.pset= gp.PrimitiveSetTyped("MAIN", [TypeB] * len(self.input), TypeB)

        for func_name in self.OP_B2B_func_list:
            func = getattr(OP.OP_B2B, func_name, None)
            self.pset.addPrimitive(func, TypeB, TypeB, name=func_name)

        for func_name in self.OP_BB2B_func_list:
            func = getattr(OP.OP_BB2B, func_name, None)
            self.pset.addPrimitive(func, [TypeB, TypeB], TypeB, name=func_name)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("MV_Root", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=1)
        self.toolbox.register("MV_Root", tools.initIterate, creator.MV_Root, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.MV_Root)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

    def generate_MV_Root(self):
        self.individuals_code = self.toolbox.population(n=self.population_size)
        self.individuals_code, self.individuals_str = change_name(self.individuals_code, self.input)


if __name__ == "__main__":
    MP_Seed = ['M_ts_mean_left_neighbor(M_O, 5, -1)', 'M_ts_mean_right_neighbor(M_C, 10, 1)']
    mp_root = MP_Root(MP_Seed)
    mp_root.generate_toolbox()
    mp_root.generate_MP_Root()
