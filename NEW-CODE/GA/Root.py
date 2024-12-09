
from GA_tools import *
from OP import *
import OP
import torch


class MP_Root:
    def __init__(self, MP_Seed: [str], population_size=10):
        self.input = MP_Seed
        self.population_size = population_size
        self.int_values = [torch.tensor(i, dtype=torch.int) for i in [2, 3, 5, 8, 10, 30]]
        self.OP_A2A_func_list = ['D_cs_rank', 'D_cs_scale', 'D_cs_zscore']
        self.OP_AA2A_func_list = ['D_cs_norm_spread', 'D_cs_cut', 'D_cs_regress_res']
        self.OP_AF2A_func_list = ['D_ts_to_max', 'D_ts_to_min', 'D_ts_to_mean', 'D_ts_norm', 'D_ts_harmonic_mean']
        self.OP_B2A_func_list = ['D_Minute_std', 'D_Minute_mean']
        self.OP_BB2A_func_list = ['D_Minute_weight_mean']
        self.OP_B2B_func_list = ['M_cs_rank', 'M_cs_scale', 'M_cs_zscore', 'M_ts_pctchg']
        self.OP_BB2B_func_list = ['M_at_div', 'M_at_sign', 'M_cs_cut', 'M_cs_norm_spread']
        self.OP_AF2C_func_list = ["Dmask_min", "Dmask_max", "Dmask_middle", "Dmask_mean_plus_std", "Dmask_mean_sub_std"]
        self.OP_B2D_func_list = ['Mmask_mean_plus_std', 'Mmask_mean_sub_std']
        self.OP_BF2D_func_list = ['Mmask_rolling_plus', 'Mmask_rolling_sub']
    def generate_toolbox(self):
        self.pset= gp.PrimitiveSetTyped("MAIN", [TypeA] * len(self.input), TypeA)
        for func_name in self.OP_A2A_func_list:
            func = getattr(OP.OP_A2A, func_name, None)
            self.pset.addPrimitive(func, TypeA, TypeA, name=func_name)

        for func_name in self.OP_AA2A_func_list:
            func = getattr(OP.OP_AA2A, func_name, None)
            self.pset.addPrimitive(func, [TypeA, TypeA], TypeA, name=func_name)


        for func_name in self.OP_AF2A_func_list:
            func = getattr(OP.OP_AF2A, func_name, None)
            self.pset.addPrimitive(func, [TypeA, TypeF], TypeA, name=func_name)

        for func_name in self.OP_B2A_func_list:
            func = getattr(OP.OP_B2A, func_name, None)
            self.pset.addPrimitive(func,TypeB, TypeA, name=func_name)

        for func_name in self.OP_BB2A_func_list:
            func = getattr(OP.OP_BB2A, func_name, None)
            self.pset.addPrimitive(func, [TypeB, TypeB], TypeA, name=func_name)


        for func_name in self.OP_B2B_func_list:
            func = getattr(OP.OP_B2B, func_name, None)
            self.pset.addPrimitive(func, TypeB, TypeB, name=func_name)

        for func_name in self.OP_BB2B_func_list:
            func = getattr(OP.OP_BB2B, func_name, None)
            self.pset.addPrimitive(func, [TypeB, TypeB], TypeB, name=func_name)


        for func_name in self.OP_AF2C_func_list:
            func = getattr(OP.OP_AF2C, func_name, None)
            self.pset.addPrimitive(func, [TypeA, TypeF], TypeC, name=func_name)

        for func_name in self.OP_B2D_func_list:
            func = getattr(OP.OP_B2D, func_name, None)
            self.pset.addPrimitive(func, TypeB, TypeD, name=func_name)

        for func_name in self.OP_BF2D_func_list:
            func = getattr(OP.OP_BF2D, func_name, None)
            self.pset_D.addPrimitive(func, [TypeB, TypeF], TypeD, name=func_name)


        for constant in self.int_values:
            self.pset.addTerminal(constant, int)

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
