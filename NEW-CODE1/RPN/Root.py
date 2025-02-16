import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

from ToolsGA.GA_tools import *
from OP import *
from GA.Config import Root_Config as Config


class MP_Root:
    def __init__(self, MP_Seed: list[str], population_size=Config.default_population):
        self.input = MP_Seed
        self.population_size = population_size
        self.OP_B2B_func_list = ['M_cs_rank', 'M_cs_scale', 'M_cs_zscore']
        self.OP_BB2B_func_list = ['M_at_div']
    def generate_toolbox(self):
        self.pset= gp.PrimitiveSetTyped("MAIN", [TypeB] * len(self.input), TypeB)

        for func_name in self.OP_B2B_func_list:
            func = getattr(OP_B2B, func_name, None)
            self.pset.addPrimitive(func, [TypeB], TypeB, name=func_name)

        for func_name in self.OP_BB2B_func_list:
            func = getattr(OP_BB2B, func_name, None)
            self.pset.addPrimitive(func, [TypeB, TypeB], TypeB, name=func_name)


        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("MP_Root", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_ = Config.min_depth, max_ = Config.max_depth)  # 树的深度按需求改
        self.toolbox.register("MP_Root", tools.initIterate, creator.MP_Root, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.MP_Root)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

    def generate_MP_Root(self):
        self.generate_toolbox()
        self.individuals_code = self.toolbox.population(n=self.population_size)
        self.individuals_code, self.individuals_str = change_name(self.individuals_code, self.input)


class MV_Root:
    def __init__(self, MV_Seed: list[str], population_size=Config.default_population):
        self.input = MV_Seed
        self.population_size = population_size
        self.OP_B2B_func_list = ['M_cs_rank', 'M_cs_scale', 'M_cs_zscore']
        self.OP_BB2B_func_list = ['M_at_div']

    def generate_toolbox(self):
        self.pset= gp.PrimitiveSetTyped("MAIN", [TypeB] * len(self.input), TypeB)

        for func_name in self.OP_B2B_func_list:
            func = getattr(OP_B2B, func_name, None)
            self.pset.addPrimitive(func, [TypeB], TypeB, name=func_name)

        for func_name in self.OP_BB2B_func_list:
            func = getattr(OP_BB2B, func_name, None)
            self.pset.addPrimitive(func, [TypeB, TypeB], TypeB, name=func_name)

        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("MV_Root", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_ = Config.min_depth, max_ = Config.max_depth)
        self.toolbox.register("MV_Root", tools.initIterate, creator.MV_Root, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.MV_Root)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

    def generate_MV_Root(self):
        self.generate_toolbox()
        self.individuals_code = self.toolbox.population(n=self.population_size)
        self.individuals_code, self.individuals_str = change_name(self.individuals_code, self.input)

class DP_Root:
    def __init__(self, DP_Seed: list[str], population_size=Config.default_population):
        self.input = DP_Seed
        self.population_size = population_size
        self.OP_AA2A_func_list = ['D_at_div']
        self.OP_AF2A_func_list = ['D_ts_pctchg','D_ts_norm']

    def generate_toolbox(self):
        self.pset= gp.PrimitiveSetTyped("MAIN",[TypeA] * len(self.input), TypeA)
        for func_name in self.OP_AA2A_func_list:
            func = getattr(OP_AA2A, func_name, None)
            self.pset.addPrimitive(func, [TypeA, TypeA], TypeA, name=func_name)

        for func_name in self.OP_AF2A_func_list:
            func = getattr(OP_AF2A, func_name, None)
            self.pset.addPrimitive(func, [TypeA, TypeF], TypeA, name=func_name)

        for constant_value in Config.default_int:
            self.pset.addTerminal(constant_value,TypeF)

        self.pset.addPrimitive(OP_Closure.id_int, [TypeF], TypeF, name='id_int')
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("DP_Root", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_ = Config.min_depth, max_ = Config.max_depth)
        self.toolbox.register("DP_Root", tools.initIterate, creator.DP_Root, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.DP_Root)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

    def generate_DP_Root(self):
        self.generate_toolbox()
        self.individuals_code = self.toolbox.population(n=self.population_size)
        self.individuals_code, self.individuals_str = change_name(self.individuals_code, self.input)

class DV_Root:
    def __init__(self, DV_Seed: list[str], population_size=Config.default_population):
        self.input = DV_Seed
        self.population_size = population_size
        self.OP_AA2A_func_list = ['D_at_div']
        self.OP_AF2A_func_list = ['D_ts_pctchg','D_ts_norm']

    def generate_toolbox(self):
        self.pset= gp.PrimitiveSetTyped("MAIN",[TypeA] * len(self.input), TypeA)
        for func_name in self.OP_AA2A_func_list:
            func = getattr(OP_AA2A, func_name, None)
            self.pset.addPrimitive(func, [TypeA, TypeA], TypeA, name=func_name)

        for func_name in self.OP_AF2A_func_list:
            func = getattr(OP_AF2A, func_name, None)
            self.pset.addPrimitive(func, [TypeA, TypeF], TypeA, name=func_name)

        for constant_value in Config.default_int:
            self.pset.addTerminal(constant_value,TypeF)
        self.pset.addPrimitive(OP_Closure.id_int, [TypeF], TypeF, name='id_int')

        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("DV_Root", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_ = Config.min_depth, max_ = Config.max_depth)
        self.toolbox.register("DV_Root", tools.initIterate, creator.DV_Root, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.DV_Root)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

    def generate_DV_Root(self):
        self.generate_toolbox()
        self.individuals_code = self.toolbox.population(n=self.population_size)
        self.individuals_code, self.individuals_str = change_name(self.individuals_code, self.input)

if __name__ == "__main__":
    MP_Seed = ['M_ts_mean_left_neighbor(M_O, 5, -1)', 'M_ts_mean_right_neighbor(M_C, 10, 1)']
    mp_root = MP_Root(MP_Seed)
    mp_root.generate_MP_Root()

    DP_Seed = ['D_ts_mean(D_O, 5)', 'D_ts_mean(D_C, 10)']
    dp_root = DP_Root(DP_Seed)
    dp_root.generate_DP_Root()
