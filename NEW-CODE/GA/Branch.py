import torch
import pandas as pd
import numpy as np
from deap import gp, base, tools, creator
from OP import *
from GA_tools import *
from Root import *

class Branch:
    def __init__(self, mp_root: 'MP_Root', population_size: int):
        self.population_size = population_size
        self.pset = None
        self.toolbox = None
        self.int_values = [torch.tensor(i, dtype=torch.int) for i in [2, 3, 5, 8, 10, 30]]
    def generate_toolbox(self):
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        class_name = self.__class__.__name__
        creator.create(class_name, gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=1)
        self.toolbox.register(class_name, tools.initIterate, getattr(creator,class_name), self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, getattr(self.toolbox,class_name))
        self.toolbox.register("compile", gp.compile, pset=self.pset)

    def generate_population(self):
        self.individuals_code = self.toolbox.population(n=self.population_size)
        self.individuals_code, self.individuals_str = change_name(self.individuals_code, self.input)


class M_Branch_MP2D(Branch):
    def __init__(self, mp_root: 'MP_Root', population_size: int):
        super().__init__(mp_root, population_size)
        self.input = mp_root
    def add_primitive(self):
        self.pset = gp.PrimitiveSetTyped("MAIN",[TypeB], TypeD)
        # 添加TypeB
        for root in self.input():
            self.pset.addTerminal(root,TypeB)      
        # 添加TypeF
        for constant in self.int_values():
            self.pset.addTerminal(constant,TypeF)
        self.pset.addPrimitive(OP_Closure.id_int(),TypeF,TypeF)
        # 添加primitive
        op = OP_BF2D
        for name in op.func_list:
            func = getattr(op, name)
            self.pset.addPrimitive(func, [TypeB,TypeF], TypeD)
        op = OP_B2D
        for name in op.func_list:
            func = getattr(op, name)
            self.pset.addPrimitive(func, TypeB, TypeD)
        # 创建工具箱
        super().generate_toolbox()
    def run(self):
        self.add_primitive()
        self.generate.population()
class M_Branch_MPDP2D(Branch):
    def __init__(self, mp_root: 'MP_Root', dp_root:'DP_Root',population_size: int):
        super().__init__(mp_root, population_size)
        self.input1 = mp_root
        self.input2 = dp_root
    def add_primitive(self):
        self.pset = gp.PrimitiveSetTyped("MAIN",[TypeB,TypeA], TypeD)
        # 添加TypeB
        for root in self.input1():
            self.pset.addTerminal(root,TypeB)      
        # 添加TypeA
        for root in self.input2():
            self.pset.addTerminal(root,TypeA)      
        # 添加primitive
        op = OP_BA2D
        for name in op.func_list:
            func = getattr(op, name)
            self.pset.addPrimitive(func, [TypeB,TypeA], TypeD)
        # 创建工具箱
        super().generate_toolbox()
    def run(self):
        self.add_primitive()
        self.generate.population()
class M_Branch_MV2D(Branch):
    def __init__(self, mv_root: 'MV_Root', population_size: int):
        super().__init__(mv_root, population_size)
        self.input = mv_root
    def add_primitive(self):
        self.pset = gp.PrimitiveSetTyped("MAIN",[TypeB], TypeD)
        # 添加TypeB
        for root in self.input():
            self.pset.addTerminal(root,TypeB)      
        # 添加TypeF
        for constant in self.int_values():
            self.pset.addTerminal(constant,TypeF)
        self.pset.addPrimitive(OP_Closure.id_int(),TypeF,TypeF)
        # 添加primitive
        op = OP_BF2D
        for name in op.func_list:
            func = getattr(op, name)
            self.pset.addPrimitive(func, [TypeB,TypeF], TypeD)
        op = OP_B2D
        for name in op.func_list:
            func = getattr(op, name)
            self.pset.addPrimitive(func, TypeB, TypeD)
        # 创建工具箱
        super().generate_toolbox()
    def run(self):
        self.add_primitive()
        self.generate.population()
class M_Branch_MVDV2D(Branch):
    def __init__(self, mv_root: 'MV_Root', dv_root:'DV_Root',population_size: int):
        super().__init__(mv_root, population_size)
        self.input1 = mv_root
        self.input2 = dv_root
    def add_primitive(self):
        self.pset = gp.PrimitiveSetTyped("MAIN",[TypeB,TypeA], TypeD)
        # 添加TypeB
        for root in self.input1():
            self.pset.addTerminal(root,TypeB)      
        # 添加TypeA
        for root in self.input2():
            self.pset.addTerminal(root,TypeA)      
        # 添加primitive
        op = OP_BA2D
        for name in op.func_list:
            func = getattr(op, name)
            self.pset.addPrimitive(func, [TypeB,TypeA], TypeD)
        # 创建工具箱
        super().generate_toolbox()
    def run(self):
        self.add_primitive()
        self.generate.population()
