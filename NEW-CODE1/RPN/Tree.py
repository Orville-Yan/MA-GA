import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

from ToolsGA.GA_tools import *
from OP import *

class config:
    default_population = 10
    min_depth = 1
    max_depth = 1

class Tree:
    def __init__(self, subtrees: list[str], population_size=config.default_population):#subtree
        self.subtrees = subtrees
        self.population_size = population_size

        self.OP_AF2A_func_list = ['D_ts_mean', 'D_ts_harmonic_mean', 'D_ts_std']

    def generate_toolbox(self):
        self.pset = gp.PrimitiveSetTyped("MAIN", [TypeA] * len(self.subtrees), TypeA)

        for func_name in self.OP_AF2A_func_list:
            func = getattr(OP_AF2A, func_name)
            self.pset.addPrimitive(func, [TypeA, TypeF], TypeA, name=func_name)

        for constant_value in [2, 3, 5, 10, 20]:
            self.pset.addTerminal(constant_value, TypeF)

        self.pset.addPrimitive(OP_Closure.id_int, [TypeF], TypeF, name='id_int')

        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Tree", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=config.min_depth, max_=config.max_depth)
        self.toolbox.register("Tree", tools.initIterate, creator.Tree, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.Tree)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

    def generate_population(self):
        self.individuals_code = self.toolbox.population(n=self.population_size)
        self.individuals_code, self.individuals_str = change_name(self.individuals_code, self.subtrees)

    def generate_tree(self):
        self.generate_toolbox()
        self.generate_population()

if __name__ == '__main__':
    Subtrees=['at_div(open,close)','at_div(high,low)','at_sign(at_sub(high,low))']
    tree=Tree(Subtrees)
    tree.generate_tree()
    print(tree.individuals_str)
