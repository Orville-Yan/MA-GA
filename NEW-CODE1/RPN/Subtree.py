import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

from ToolsGA.GA_tools import *
from OP import *

class SubtreeBase:
    def __init__(self, population_size=config.default_population):
        self.population_size = population_size
        self.pset = None
        self.toolbox = None

    def generate_toolbox(self, class_name, pset):
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))

        fitness_class_name = f"FitnessMax_{class_name}"
        individual_class_name = f"Subtree_{class_name}"

        if not hasattr(creator, fitness_class_name):
            creator.create(fitness_class_name, base.Fitness, weights=(1.0,))
        if not hasattr(creator, individual_class_name):
            creator.create(individual_class_name, gp.PrimitiveTree, fitness=getattr(creator, fitness_class_name), pset=pset)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=1)
        self.toolbox.register("individual", tools.initIterate, getattr(creator, individual_class_name), self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=pset)

    def generate_population(self):
        self.individuals_code = self.toolbox.population(n=self.population_size)

    def add_primitive(self):
        raise NotImplementedError

    def run(self):
        self.add_primitive()
        self.generate_population()

class SubtreeWithMask(SubtreeBase):
    def __init__(self, input1, input2, population_size=config.default_population):
        super().__init__(population_size)
        self.input1 = input1
        self.input2 = input2
        self.OP_BD2A_func_list = ['D_Minute_area_mean', 'D_Minute_area_std', 'D_Minute_area_sum', 'D_Minute_area_prod']
        self.OP_BBD2A_func_list = ['D_Minute_area_weight_mean', 'D_Minute_area_corr', 'D_Minute_area_rankcorr',
                                   'D_Minute_area_bifurcate_mean', 'D_Minute_area_bifurcate_std']

    def add_primitive(self):
        self.pset = gp.PrimitiveSetTyped("MAIN_with_mask", [TypeB]*len(self.input1) + [TypeD]*len(self.input2), TypeA)

        for func_name in self.OP_BD2A_func_list:
            func = getattr(OP_BD2A, func_name)
            self.pset.addPrimitive(func, [TypeB, TypeD], TypeA, name=func_name)

        for func_name in self.OP_BBD2A_func_list:
            func = getattr(OP_BBD2A, func_name)
            self.pset.addPrimitive(func, [TypeB, TypeB, TypeD], TypeA, name=func_name)

        self.generate_toolbox("with_mask", self.pset)

    def generate_population(self):
        super().generate_population()
        self.individuals_code, self.individuals_str = change_name(self.individuals_code, self.input1 + self.input2)


class SubtreeNoMask(SubtreeBase):
    def __init__(self, input_data, population_size=config.default_population):
        super().__init__(population_size)
        self.input = input_data
        self.OP_B2A_func_list = ['D_Minute_std', 'D_Minute_mean', 'D_Minute_trend']
        self.OP_BB2A_func_list = ['D_Minute_corr', 'D_Minute_weight_mean']

    def add_primitive(self):
        self.pset = gp.PrimitiveSetTyped("MAIN_no_mask", [TypeB]*len(self.input), TypeA)

        for func_name in self.OP_B2A_func_list:
            func = getattr(OP_B2A, func_name)
            self.pset.addPrimitive(func, [TypeB], TypeA, name=func_name)

        for func_name in self.OP_BB2A_func_list:
            func = getattr(OP_BB2A, func_name)
            self.pset.addPrimitive(func, [TypeB, TypeB], TypeA, name=func_name)

        self.generate_toolbox("no_mask", self.pset)

    def generate_population(self):
        super().generate_population()
        self.individuals_code, self.individuals_str = change_name(self.individuals_code, self.input)

if __name__ == "__main__":
    input_data = ['open','high']
    mask = ['low']

    subtree_with_mask = SubtreeWithMask(input_data, mask, population_size=config.default_population)
    subtree_with_mask.run()
    print("With Mask:", subtree_with_mask.individuals_str)

    subtree_no_mask = SubtreeNoMask(input_data, population_size=config.default_population)
    subtree_no_mask.run()
    print("No Mask:", subtree_no_mask.individuals_str)
