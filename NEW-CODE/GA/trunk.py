# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 00:01:40 2024

@author: 朱培元
"""
import sys
sys.path.append('..')

from Tools.GA_tools import *
from OP import *

class MP_Trunk:
    def __init__(self,MP_Root: list[str], population_size=100):
        self.input=MP_Root
        self.population_size=population_size
        self.OP_BB2B_func_list=['M_at_add', 'M_at_sub', 'M_at_div', 'M_at_sign', 'M_cs_cut', 'M_cs_umr', 'M_at_prod', 'M_cs_norm_spread']#你这个class需要用到的算子类别的func_list
        self.OP_BA2B_func_list=['M_toD_standard']
        self.OP_A2A_func_list=['D_at_abs', 'D_cs_rank', 'D_cs_scale', 'D_cs_zscore', 'D_cs_harmonic_mean', 'D_cs_demean','D_cs_winsor']
        self.OP_AE2A_func_list=['D_cs_demean_industry', 'D_cs_industry_neutra']
        self.OP_AA2A_func_list=['D_cs_norm_spread', 'D_cs_cut', 'D_cs_regress_res', 'D_at_add', 'D_at_sub', 'D_at_div','D_at_prod', 'D_at_mean']
        self.OP_AG2A_func_list=['D_cs_edge_flip']
        self.OP_AAF2A_func_list=['D_ts_corr', 'D_ts_rankcorr', 'D_ts_regress_res', 'D_ts_weight_mean', 'D_ts_regress']
        self.OP_AF2A_func_list=['D_ts_max', 'D_ts_min', 'D_ts_delay', 'D_ts_delta', 'D_ts_pctchg',
                          'D_ts_mean', 'D_ts_harmonic_mean', 'D_ts_std', 'D_ts_to_max',
                          'D_ts_to_min', 'D_ts_to_mean', 'D_ts_max_to_min', 
                          'D_ts_norm', 'D_ts_detrend'
                          ]
        self.OP_BD2A_func_list=['D_Minute_area_mean', 'D_Minute_area_std', 'D_Minute_area_sum', 'D_Minute_area_prod']
        self.OP_BBD2A_func_list=['D_Minute_area_weight_mean', 'D_Minute_area_corr', 'D_Minute_area_rankcorr',
                          'D_Minute_area_bifurcate_mean', 'D_Minute_area_bifurcate_std']

    def generate_toolbox(self):
        self.pset=gp.PrimitiveSetTyped("MAIN", [TypeB] * len(self.input), TypeB)
           
        #注册需要用到的primitives和terminals
        for func_name in self.OP_BB2B_func_list:
            func = getattr(OP_BB2B, func_name) 
            self.pset.addPrimitive(func, [TypeB, TypeB], TypeB, name=func_name)
            
        for func_name in self.OP_BA2B_func_list:
            func = getattr(OP_BA2B, func_name)
            self.pset.addPrimitive(func, [TypeB, TypeA], TypeB, name=func_name)
            
        for func_name in self.OP_A2A_func_list:
            func = getattr(OP_A2A, func_name)
            self.pset.addPrimitive(func, [TypeA], TypeA, name=func_name)
            
        for func_name in self.OP_AE2A_func_list:
            func = getattr(OP_AE2A, func_name)
            self.pset.addPrimitive(func, [TypeA, TypeE], TypeA, name=func_name)
            
        for func_name in self.OP_AA2A_func_list:
            func = getattr(OP_AA2A, func_name)
            self.pset.addPrimitive(func, [TypeA, TypeA], TypeA, name=func_name)
            
        for func_name in self.OP_AG2A_func_list:
            func = getattr(OP_AG2A, func_name)
            self.pset.addPrimitive(func, [TypeA, TypeG], TypeA, name=func_name)
            
        for func_name in self.OP_AAF2A_func_list:
            func = getattr(OP_AAF2A, func_name)
            self.pset.addPrimitive(func, [TypeA, TypeA,TypeF], TypeA, name=func_name)
            
        for func_name in self.OP_AF2A_func_list:
            func = getattr(OP_AF2A, func_name)
            self.pset.addPrimitive(func, [TypeA, TypeF], TypeA, name=func_name)
            
        for func_name in self.OP_BD2A_func_list:
            func = getattr(OP_BD2A, func_name)
            self.pset.addPrimitive(func, [TypeB, TypeD], TypeA, name=func_name)
            
        for func_name in self.OP_BBD2A_func_list:
            func = getattr(OP_BBD2A, func_name)
            self.pset.addPrimitive(func, [TypeB, TypeB, TypeD], TypeA, name=func_name)
        #......
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("MP_Trunk", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3) #树的深度按需求改
        self.toolbox.register("MP_Trunk", tools.initIterate, creator.MP_Trunk, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.MP_Trunk)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        
    def generate_MP_Root(self):
        self.individuals_code = self.toolbox.population(n=self.population_size)
        self.individuals_code, self.individuals_str = change_name(self.individuals_code, self.input)

if __name__ == '__main__':
    MP_Root=['at_div(open,close)','at_div(high,low)','at_sign(at_sub(high,low))']
    mp_trunk=MP_Trunk(MP_Root)
    mp_trunk.generate_toolbox()
    mp_trunk.generate_MP_Root()
