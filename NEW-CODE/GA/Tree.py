import sys
sys.path.append('..')

from deap import gp, creator, base, tools
import numpy as np
from Tools.GA_tools import *
from OP.ToA import *
from OP.Others import *
#from Subtree import Subtree

class Tree:
    def __init__(self, subtrees: list[str], population_size=10):#subtree
        self.subtrees = subtrees
        self.population_size = population_size
        self.OP_A2A_func_list = ['D_cs_rank', 'D_cs_scale', 'D_cs_zscore', 'D_cs_harmonic_mean', 'D_cs_demean',
                          'D_cs_winsor']
        self.OP_AF2A_func_list = ['D_ts_max', 'D_ts_min', 'D_ts_delay', 'D_ts_delta', 'D_ts_pctchg',
                          'D_ts_mean', 'D_ts_harmonic_mean', 'D_ts_std', 'D_ts_to_max',
                          'D_ts_to_min', 'D_ts_to_mean', 'D_ts_max_to_min', 'D_ts_maxmin_norm',
                          'D_ts_norm', 'D_ts_detrend']

    def generate_toolbox(self):
        self.pset = gp.PrimitiveSetTyped("MAIN", [TypeA] * len(self.subtrees), TypeA)

        for func_name in self.OP_A2A_func_list:
            func = getattr(OP_A2A, func_name)
            self.pset.addPrimitive(func, [TypeA], TypeA, name=func_name)

        for func_name in self.OP_AF2A_func_list:
            func = getattr(OP_AF2A, func_name)
            self.pset.addPrimitive(func, [TypeA, TypeF], TypeA, name=func_name)


        int_values = [int(i) for i in [2, 3, 5, 10, 20, 60]]
        for constant_value in int_values:
            self.pset.addTerminal(constant_value, TypeF)

        self.pset.addPrimitive(OP_Closure.id_int, [TypeF], TypeF, name='id_int')


        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Tree", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=1)
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
    tree.generate_toolbox()
    tree.generate_tree()
    #print(tree.individuals_str)
