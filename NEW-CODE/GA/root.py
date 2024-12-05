from GA_tools import *
from OP import *
import OP

class MP_Root:
    def __init__(self, MP_Seed: [str], population_size=10):
        self.input = MP_Seed
        self.population_size = population_size
        self.OP_A2A_func_list = ['D_cs_scale', 'D_cs_zscore''D_cs_harmonic_mean']
        self.OP_AA2A_func_list = ['D_cs_norm_spread']
        self.OP_AF2A_func_list = ['D_ts_to_max','D_ts_to_min', 'D_ts_to_mean','D_ts_norm','D_ts_harmonic_mean']
        self.OP_B2B_func_list = ['M_cs_rank', 'M_cs_scale', 'M_cs_zscore','M_cs_demean']
        self.OP_BB2B_func_list = ['cs_norm_spread']
        self.OP_BA2B_func_list = ['M_toD_standard']
        self.OP_AF2C_func_list = ['Dmask_mean_plus_std', 'Dmask_mean_sub_std']
        self.OP_B2D_func_list = ['Mmask_mean_plus_std','Mmask_mean_sub_std']
        self.OP_BF2D_func_list = ['Mmask_rolling_plus','Mmask_rolling_sub']

        #self.OP_BB2B_func_list = ['at_div']  # 你这个class需要用到的算子类别的func_lists


    def generate_toolbox(self):
        self.pset_A = gp.PrimitiveSetTyped("MAIN_A", [TypeA] * len(self.input), TypeA)
        self.pset_B = gp.PrimitiveSetTyped("MAIN_B", [TypeB] * len(self.input), TypeB)
        self.pset_C = gp.PrimitiveSetTyped("MAIN_C", [TypeC] * len(self.input), TypeC)
        self.pset_D = gp.PrimitiveSetTyped("MAIN_D", [TypeD] * len(self.input), TypeD)
        self.pset_E = gp.PrimitiveSetTyped("MAIN_E", [TypeE] * len(self.input), TypeE)
        self.pset_F = gp.PrimitiveSetTyped("MAIN_F", [TypeF] * len(self.input), TypeF)

        for func_name in self.OP_A2A_func_list:
            func = getattr(OP.OP_A2A, func_name, None)
            self.pset_A.addPrimitive(func,TypeA, TypeA, name=func_name)
            self.pset_A.addTerminal(func,ret_type=TypeA,name=func_name)

        for func_name in self.OP_AA2A_func_list:
            func = getattr(OP.OP_AA2A, func_name, None)
            self.pset_A.addPrimitive(func, [TypeA, TypeA], TypeA, name=func_name)
            self.pset_A.addTerminal(func,ret_type=TypeA,name=func_name)

        for func_name in self.OP_AF2A_func_list:
            func = getattr(OP.OP_AF2A, func_name, None)
            self.pset_A.addPrimitive(func, [TypeA, TypeF], TypeA, name=func_name)
            self.pset_A.addTerminal(func,ret_type=TypeA,name=func_name)

        for func_name in self.OP_B2B_func_list:
            func = getattr(OP.OP_B2B, func_name, None)
            self.pset_B.addPrimitive(func, TypeB, TypeB, name=func_name)
            self.pset_B.addTerminal(func,ret_type=TypeB,name=func_name)

        for func_name in self.OP_BB2B_func_list:
            func = getattr(OP.OP_BB2B, func_name, None)
            self.pset_B.addPrimitive(func, [TypeB,TypeB], TypeB, name=func_name)
            self.pset_B.addTerminal(func,ret_type=TypeB,name=func_name)

        for func_name in self.OP_BA2B_func_list:
            func = getattr(OP.OP_BA2B, func_name, None)
            self.pset_B.addPrimitive(func, [TypeB, TypeA], TypeB, name=func_name)
            self.pset_B.addTerminal(func,ret_type=TypeB,name=func_name)



        for func_name in self.OP_AF2C_func_list:
            func = getattr(OP.OP_AF2C, func_name, None)
            self.pset_C.addPrimitive(func, [TypeA, TypeF], TypeC, name=func_name)
            self.pset_C.addTerminal(func,ret_type=TypeC,name=func_name)

        for func_name in self.OP_B2D_func_list:
            func = getattr(OP.OP_B2D, func_name, None)
            self.pset_D.addPrimitive(func, TypeB, TypeD, name=func_name)
            self.pset_D.addTerminal(func,ret_type=TypeD,name=func_name)

        for func_name in self.OP_BF2D_func_list:
            func = getattr(OP.OP_BF2D, func_name, None)
            self.pset_D.addPrimitive(func, [TypeB, TypeF], TypeD, name=func_name)
            self.pset_D.addTerminal(func,ret_type=TypeD,name=func_name)


        # ......

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("MP_Root", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=1)  # 树的深度按需求改
        self.toolbox.register("MP_Root", tools.initIterate, creator.MP_Root, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.MP_Root)
        self.toolbox.register("compile", gp.compile, pset=self.pset)


    def generate_MP_Root(self):
        self.individuals_code = self.toolbox.population(n=self.population_size)
        self.individuals_code, self.individuals_str = change_name(self.individuals_code, self.input)


MP_Seed = ['M_ts_mean_left_neighbor(M_O, 5, -1)', 'M_ts_mean_right_neighbor(M_C, 10, 1)']
mp_root = MP_Root(MP_Seed)
mp_root.generate_toolbox()
mp_root.generate_MP_Root()