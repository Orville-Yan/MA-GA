import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

from ToolsGA.GA_tools import *
from OP import *

from GA.Config import Trunk_Config as Config
class Trunk:
    def __init__(self, M_Root: list[str], D_Root: list[str], Branch: list[str], ind_str: list[str],population_size=100):
        self.input1 = M_Root
        self.input2 = D_Root
        self.input3 = Branch
        self.input4 = ind_str
        self.population_size = population_size

    def add_primitive_byclass(self, op_classname):
        parts = op_classname.split('2')
        part1 = parts[0].split('_')[1]
        part2 = parts[1]

        op = globals()[op_classname]()
        for func_name in op.func_list:
            func = getattr(op, func_name)
            input_class_list = [globals()[f"Type{char}"] for char in part1]
            output_class = globals()[f"Type{part2}"]
            self.pset.addPrimitive(func, input_class_list, output_class, name=func_name)
    def generate_toolbox(self):
        self.pset = gp.PrimitiveSetTyped("MAIN", [TypeB] * len(self.input1)+ [TypeA]*len(self.input2)+ [TypeD]*len(self.input3)+ [TypeE]*len(self.input4), TypeB)

        for constant_value in Config.default_lookback:
            self.pset.addTerminal(constant_value, TypeF)

        for constant_value in Config.default_edge:
            self.pset.addTerminal(constant_value, TypeG)

        for class_name in globals()["OPclass_name_2B"]:
            self.add_primitive_byclass(class_name)

        self.pset.addPrimitive(OP_Closure.id_int, [TypeF], TypeF, name='id_int')
        self.pset.addPrimitive(OP_Closure.id_industry, [TypeE], TypeE, name='id_industry')
        self.pset.addPrimitive(OP_Closure.id_float, [TypeG], TypeG, name='id_float')
        self.pset.addPrimitive(OP_Closure.id_tensor,[TypeA],TypeA,name='id_tensor')
        if not hasattr(creator,"FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Trunk", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=Config.min_depth, max_=Config.max_depth)  # 树的深度按需求改
        self.toolbox.register("Trunk", tools.initIterate, creator.Trunk, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.Trunk)

    def generate_population(self):
        self.individuals_code = self.toolbox.population(n=self.population_size)
        self.individuals_code, self.individuals_str = change_name(self.individuals_code, self.input1+self.input2+self.input3+self.input4)
    def run(self):
        self.generate_toolbox()
        self.generate_population()


if __name__ == '__main__':

    M_Root = ['at_div(open,close)', 'at_div(high,low)', 'at_sign(at_sub(high,low))']
    op_A = ['at_mean(open,close)']
    op_D = ['mask_max(high)']
    op_E = ['mask_max(high)']
    #industry_used = data_reader.get_barra([2016, 2017])
    mp_trunk = Trunk(M_Root,op_A,op_D,op_E)
    mp_trunk.run()
