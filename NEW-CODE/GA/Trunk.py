import sys
sys.path.append('..')

from ToolsGA.GA_tools import *
from OP.ToB import *
from OP.ToA import *
from OP.Others import *
from deap import gp, creator, base, tools
class Trunk:
    def __init__(self, M_Root: list[str], D_Root, Branch, ind_str,population_size=100):
        self.input1 = M_Root
        self.input2 = D_Root
        self.input3 = Branch
        self.input4 = ind_str
        self.population_size = population_size
        #self.industry=industry_used
        self.OP_BB2B_func_list = ['M_at_add', 'M_at_sub', 'M_at_div', 'M_at_sign', 'M_cs_cut', 'M_cs_umr', 'M_at_prod',
                                  'M_cs_norm_spread']  # 你这个class需要用到的算子类别的func_list
        self.OP_BA2B_func_list = ['M_toD_standard']
        self.OP_A2A_func_list = ['D_at_abs', 'D_cs_rank', 'D_cs_scale', 'D_cs_zscore', 'D_cs_harmonic_mean',
                                 'D_cs_demean', 'D_cs_winsor']
        self.OP_AE2A_func_list = ['D_cs_demean_industry', 'D_cs_industry_neutra']
        self.OP_AA2A_func_list = ['D_cs_norm_spread', 'D_cs_cut', 'D_cs_regress_res', 'D_at_add', 'D_at_sub',
                                  'D_at_div', 'D_at_prod', 'D_at_mean']
        self.OP_AG2A_func_list = ['D_cs_edge_flip']
        self.OP_AAF2A_func_list = ['D_ts_corr', 'D_ts_rankcorr', 'D_ts_regress_res', 'D_ts_weight_mean', 'D_ts_regress']
        self.OP_AF2A_func_list = ['D_ts_max', 'D_ts_min', 'D_ts_delay', 'D_ts_delta', 'D_ts_pctchg',
                                  'D_ts_mean', 'D_ts_harmonic_mean', 'D_ts_std', 'D_ts_to_max',
                                  'D_ts_to_min', 'D_ts_to_mean', 'D_ts_max_to_min',
                                  'D_ts_norm', 'D_ts_detrend'
                                  ]
        self.OP_BD2A_func_list = ['D_Minute_area_mean', 'D_Minute_area_std', 'D_Minute_area_sum', 'D_Minute_area_prod']
        self.OP_BBD2A_func_list = ['D_Minute_area_weight_mean', 'D_Minute_area_corr', 'D_Minute_area_rankcorr',
                                   'D_Minute_area_bifurcate_mean', 'D_Minute_area_bifurcate_std']

    def generate_toolbox(self):
        self.pset = gp.PrimitiveSetTyped("MAIN", [TypeB] * len(self.input1)+ [TypeA]*len(self.input2)+ [TypeD]*len(self.input3)+ [TypeE]*len(self.input4), TypeB)
        
        int_values = [int(i) for i in [2, 3, 5, 8, 10, 30]]
        for constant_value in [2,3,5,10,20]:
            self.pset.addTerminal(constant_value,TypeF)
        # 注册需要用到的primitives和terminals
        for func_name in self.OP_BB2B_func_list:
            func = getattr(OP_BB2B, func_name)
            self.pset.addPrimitive(func, [TypeB, TypeB], TypeB, name=func_name)

        for func_name in self.OP_BA2B_func_list:
            func = getattr(OP_BA2B, func_name)
            self.pset.addPrimitive(func, [TypeB, TypeA], TypeB, name=func_name)

        for func_name in self.OP_A2A_func_list:
            func = getattr(OP_A2A, func_name)
            self.pset.addPrimitive(func, [TypeA], TypeA, name=func_name)

        # for func_name in self.OP_AE2A_func_list:
        #     func = getattr(OP_AE2A, func_name)
        #     self.pset.addPrimitive(func, [TypeA, TypeE], TypeA, name=func_name)

        for func_name in self.OP_AA2A_func_list:
            func = getattr(OP_AA2A, func_name)
            self.pset.addPrimitive(func, [TypeA, TypeA], TypeA, name=func_name)

        for func_name in self.OP_AG2A_func_list:
            func = getattr(OP_AG2A, func_name)
            self.pset.addPrimitive(func, [TypeA, TypeG], TypeA, name=func_name)

        for func_name in self.OP_AAF2A_func_list:
            func = getattr(OP_AAF2A, func_name)
            self.pset.addPrimitive(func, [TypeA, TypeA, TypeF], TypeA, name=func_name)

        for func_name in self.OP_AF2A_func_list:
            func = getattr(OP_AF2A, func_name)
            self.pset.addPrimitive(func, [TypeA, TypeF], TypeA, name=func_name)

        for func_name in self.OP_BD2A_func_list:
            func = getattr(OP_BD2A, func_name)
            self.pset.addPrimitive(func, [TypeB, TypeD], TypeA, name=func_name)

        for func_name in self.OP_BBD2A_func_list:
            func = getattr(OP_BBD2A, func_name)
            self.pset.addPrimitive(func, [TypeB, TypeB, TypeD], TypeA, name=func_name)
        # ......
        # self.pset.addPrimitive(OP_Closure.id_int, [TypeF], TypeF, name='id_int')
        # self.pset.addPrimitive(OP_Closure.id_tensor, [TypeB], TypeB, name='id_tensor')
        self.pset.addPrimitive(OP_Closure.id_int, [TypeF], TypeF, name='id_int')
        self.pset.addPrimitive(OP_Closure.id_industry, [TypeE], TypeE, name='id_industry')
        #self.pset.addTerminal(self.industry,TypeE)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Trunk", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3)  # 树的深度按需求改
        self.toolbox.register("Trunk", tools.initIterate, creator.Trunk, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.Trunk)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

    def generate_Trunk(self):
        self.individuals_code = self.toolbox.population(n=self.population_size)
        self.individuals_code, self.individuals_str = change_name(self.individuals_code, self.input1+self.input2+self.input3+self.input4)


if __name__ == '__main__':
    from ToolsGA.DataReader import DataReader
    data_reader = DataReader()
    DO, DH, DL, DC, DV = data_reader.get_Day_data([2016, 2017])
    print(DO.shape)
    M_Root = ['at_div(open,close)', 'at_div(high,low)', 'at_sign(at_sub(high,low))']
    op_A = ['at_mean(open,close)']
    op_D = ['mask_max(high)']
    op_E = ['mask_max(high)']
    #industry_used = data_reader.get_barra([2016, 2017])
    mp_trunk = Trunk(M_Root,op_A,op_D,op_E)
    mp_trunk.generate_toolbox()
    mp_trunk.generate_Trunk()
