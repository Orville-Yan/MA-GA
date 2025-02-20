import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

from ToolsGA.GA_tools import *
from OP import *

from GA.Config import Root_Config as Config
class Root:
    def generate_toolbox(self):
        classname = self.__class__.__name__
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create(classname,gp.PrimitiveTree,pset = self.pset)
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_= Config.min_depth, max_=Config.max_depth)
        self.toolbox.register(classname, tools.initIterate, getattr(creator, classname), self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, getattr(self.toolbox, classname))

    def generate_population(self):
        self.individuals_code = self.toolbox.population(n=self.population_size)
        self.individuals_code, self.individuals_str = change_name(self.individuals_code, self.input)

class MP_Root(Root):
    def __init__(self, MP_Seed: list[str], population_size=10):
        self.input = MP_Seed
        self.population_size = population_size

    def add_primitive(self):
        self.pset= gp.PrimitiveSetTyped("MAIN", [TypeB] * len(self.input), TypeB)

        for func_name in Config.OP_B2B_func_list:
            func = getattr(OP_B2B, func_name, None)
            self.pset.addPrimitive(func, [TypeB], TypeB, name=func_name)

        for func_name in Config.OP_BB2B_func_list:
            func = getattr(OP_BB2B, func_name, None)
            self.pset.addPrimitive(func, [TypeB, TypeB], TypeB, name=func_name)
        super().generate_toolbox()
    def run(self):
        self.add_primitive()
        self.generate_population()


class MV_Root(Root):
    def __init__(self, MV_Seed: list[str], population_size=10):
        self.input = MV_Seed
        self.population_size = population_size

    def add_primitive(self):
        self.pset= gp.PrimitiveSetTyped("MAIN", [TypeB] * len(self.input), TypeB)

        for func_name in Config.OP_B2B_func_list:
            func = getattr(OP_B2B, func_name, None)
            self.pset.addPrimitive(func, [TypeB], TypeB, name=func_name)

        for func_name in Config.OP_BB2B_func_list:
            func = getattr(OP_BB2B, func_name, None)
            self.pset.addPrimitive(func, [TypeB, TypeB], TypeB, name=func_name)
        super().generate_toolbox()
    def run(self):
        self.add_primitive()
        self.generate_population()

class DP_Root(Root):
    def __init__(self, DP_Seed: list[str], population_size=10):
        self.input = DP_Seed
        self.population_size = population_size

    def add_primitive(self):
        self.pset= gp.PrimitiveSetTyped("MAIN",[TypeA] * len(self.input), TypeA)
        for func_name in Config.OP_AA2A_func_list:
            func = getattr(OP_AA2A, func_name, None)
            self.pset.addPrimitive(func, [TypeA, TypeA], TypeA, name=func_name)

        for func_name in Config.OP_AF2A_func_list:
            func = getattr(OP_AF2A, func_name, None)
            self.pset.addPrimitive(func, [TypeA, TypeF], TypeA, name=func_name)

        for constant_value in Config.default_lookback:
            self.pset.addTerminal(constant_value,TypeF)

        self.pset.addPrimitive(OP_Closure.id_int, [TypeF], TypeF, name='id_int')
        super().generate_toolbox()
    def run(self):
        self.add_primitive()
        self.generate_population()

class DV_Root(Root):
    def __init__(self, DV_Seed: list[str], population_size=10):
        self.input = DV_Seed
        self.population_size = population_size

    def add_primitive(self):
        self.pset= gp.PrimitiveSetTyped("MAIN",[TypeA] * len(self.input), TypeA)
        for func_name in Config.OP_AA2A_func_list:
            func = getattr(OP_AA2A, func_name, None)
            self.pset.addPrimitive(func, [TypeA, TypeA], TypeA, name=func_name)

        for func_name in Config.OP_AF2A_func_list:
            func = getattr(OP_AF2A, func_name, None)
            self.pset.addPrimitive(func, [TypeA, TypeF], TypeA, name=func_name)

        for constant_value in Config.default_lookback:
            self.pset.addTerminal(constant_value,TypeF)
        self.pset.addPrimitive(OP_Closure.id_int, [TypeF], TypeF, name='id_int')
        super().generate_toolbox()


    def run(self):
        self.add_primitive()
        self.generate_population()

if __name__ == "__main__":
    MP_Seed = ['M_ts_mean_left_neighbor(M_O, 5, -1)', 'M_ts_mean_right_neighbor(M_C, 10, 1)']
    mp_root = MP_Root(MP_Seed)
    mp_root.run()
    print(mp_root.individuals_code)

    DP_Seed = ['D_ts_mean(D_O, 5)', 'D_ts_mean(D_C, 10)']
    dp_root = DP_Root(DP_Seed)
    dp_root.run( )
    print(dp_root.individuals_code)

    

