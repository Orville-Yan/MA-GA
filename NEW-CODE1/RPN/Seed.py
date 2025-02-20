import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

from ToolsGA.GA_tools import *
from OP import *
from GA.Config import Seed_Config as Config


class Seed:
    def generate_toolbox(self):
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        class_name = self.__class__.__name__
        creator.create(class_name, gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=Config.min_depth, max_=Config.max_depth)
        self.toolbox.register(class_name, tools.initIterate, getattr(creator, class_name), self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, getattr(self.toolbox, class_name))


    def generate_population(self):
        self.individuals_code = self.toolbox.population(n=self.population_size)
        self.individuals_code, self.individuals_str = change_name(self.individuals_code, self.input)


class DP_Seed(Seed):
    def __init__(self, D_OHLC, population_size=10):
        self.input = D_OHLC
        self.population_size = population_size

    def add_primitive(self):
        self.pset = gp.PrimitiveSetTyped("MAIN", [TypeA] * len(self.input), TypeA)

        for func_name in Config.OP_AF2A_func_list:
            func = getattr(OP_AF2A, func_name)
            self.pset.addPrimitive(func, [TypeA, TypeF], TypeA, name=func_name)

        for constant_value in Config.default_lookback:
            self.pset.addTerminal(constant_value, TypeF)
        self.pset.addPrimitive(OP_Closure.id_int, [TypeF], TypeF, name='id_int')

        super().generate_toolbox()

    def run(self):
        self.add_primitive()
        self.generate_population()


class DV_Seed(Seed):
    def __init__(self, D_V, population_size=10):
        self.input = D_V
        self.population_size = population_size


    def add_primitive(self):
        self.pset = gp.PrimitiveSetTyped("MAIN", [TypeA] * len(self.input), TypeA)
        for func_name in Config.    OP_AF2A_func_list:
            func = getattr(OP_AF2A, func_name)
            self.pset.addPrimitive(func, [TypeA, TypeF], TypeA, name=func_name)
        for constant_value in Config.default_lookback:
            self.pset.addTerminal(constant_value, TypeF)
        for func_name in Config.OP_AA2A_func_list:
            func = getattr(OP_AA2A, func_name)
            self.pset.addPrimitive(func, [TypeA, TypeA], TypeA, name=func_name)

        self.pset.addPrimitive(OP_Closure.id_int, [TypeF], TypeF, name='id_int')
        super().generate_toolbox()

    def run(self):
        self.add_primitive()
        self.generate_population()


class MP_Seed(Seed):
    def __init__(self, M_OHLC, population_size=10):
        self.input = M_OHLC
        self.population_size = population_size

    def add_primitive(self):
        self.pset = gp.PrimitiveSetTyped("MAIN", [TypeB] * len(self.input), TypeB)

        for constant_value in Config.default_lookback:
            self.pset.addTerminal(constant_value, TypeF)

        for func_name in Config.OP_BF2B_func_list:
            func = getattr(OP_BF2B, func_name)
            self.pset.addPrimitive(func, [TypeB, TypeF], TypeB, name=func_name)
        self.pset.addPrimitive(OP_Closure.id_int, [TypeF], TypeF, name='id_int')
        super().generate_toolbox()

    def run(self):
        self.add_primitive()
        self.generate_population()


class MV_Seed(Seed):
    def __init__(self, M_V, population_size=10):
        self.input = M_V
        self.population_size = population_size


    def add_primitive(self):
        self.pset = gp.PrimitiveSetTyped("MAIN", [TypeB] * len(self.input), TypeB)
        for constant_value in Config.default_lookback:
            self.pset.addTerminal(constant_value, TypeF)

        for func_name in Config.OP_BF2B_func_list:
            func = getattr(OP_BF2B, func_name)
            self.pset.addPrimitive(func, [TypeB, TypeF], TypeB, name=func_name)
        self.pset.addPrimitive(OP_Closure.id_int, [TypeF], TypeF, name='id_int')
        super().generate_toolbox()

    def run(self):
        self.add_primitive()
        self.generate_population()


if __name__ == "__main__":
    mp_seed = MP_Seed(['M_O', 'M_H', 'M_L', 'M_C'],10)
    mp_seed.run()
    mv_seed = MV_Seed(['M_V'],10)
    mv_seed.run()
    print("MP_Seed Individual Str: ", mp_seed.individuals_str)
    print("MP_Seed Individual Code: ", mp_seed.individuals_code)
    print("MV_Seed Individual Str: ", mv_seed.individuals_str)
    print("MV_Seed Individual Code: ", mv_seed.individuals_code)