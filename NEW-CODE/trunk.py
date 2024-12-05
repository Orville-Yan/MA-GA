# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 00:01:40 2024

@author: 朱培元
"""
from GA_tools import *
from OP import *
from OP.ToB import OP_BB2B

class MP_Root:
    def __init__(self,MP_Seed:[str],population_size=100):
        self.input=MP_Seed
        self.population_size=population_size
        self.OP_BB2B_func_list=['M_at_add', 'M_at_sub', 'M_at_div', 'M_at_sign', 'M_cs_cut', 'M_cs_umr', 'M_at_prod', 'M_cs_norm_spread']#你这个class需要用到的算子类别的func_list
        
    def generate_toolbox(self):
        self.pset=gp.PrimitiveSetTyped("MAIN", [TypeB] * len(self.input), TypeB)
            
        #注册需要用到的primitives和terminals
        for func_name in self.OP_BB2B_func_list:
            func = getattr(OP_BB2B, func_name) 
            self.pset.addPrimitive(func, [TypeB, TypeB], TypeB, name=func_name)
        #......
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("MP_Root", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3) #树的深度按需求改
        self.toolbox.register("MP_Root", tools.initIterate, creator.MP_Root, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.MP_Root)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        
    def generate_MP_Root(self):
        self.individuals_code = self.toolbox.population(n=self.population_size)
        self.individuals_code, self.individuals_str = change_name(self.individuals_code, self.input)

MP_Seed=['at_div(open,close)','at_div(high,low)','at_sign(at_sub(high,low))']
mp_root=MP_Root(MP_Seed)
mp_root.generate_toolbox()
mp_root.generate_MP_Root()