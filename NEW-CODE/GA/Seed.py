import sys
sys.path.append('..')

from Tools.GA_tools import *
from OP import *

class other:#看到的其他算子，不用管
    def break_high(D_C):#创新高
        return OP_AF2A.D_is_max(D_C == OP_AF2A.D_ts_max(D_C,lookback = 240))
    
    def calculate_PATV(M_v):#持续异常交易量
        def previous_M_v_mean(M_v):
 
            previous_day_mean = OP_Basic.nanmean(OP_B2B.M_ts_delay(M_v, 1))
    
            return previous_day_mean
    # Step1: 计算ATV_d = 分钟交易量 / 前一日分钟交易量的均值
        ATV_d = OP_BB2B.M_at_div(M_v, previous_M_v_mean(M_v))
    
    # Step2: 计算ATV_d在截面上的排名分位数rank_ATV_d
        rank_ATV_d = OP_B2B.M_cs_rank(ATV_d)
    
    # Step3: 计算PATV_d = (mean_rank_ATV_d / std_rank_ATV_d) + kurtosis_rank_ATV_d
        mean_rank = torch.mean(rank_ATV_d, dim=-1, keepdim=True)
        std_rank = torch.std(rank_ATV_d, dim=-1, keepdim=True)
    
    # 处理标准差为0的情况
        std_rank = torch.where(std_rank == 0, torch.tensor(float('nan')), std_rank)
    
    # 计算峰度
        def calculate_kurtosis(tensor, dim=-1):
            tensor = tensor.masked_fill(torch.isnan(tensor), 0)
            mean = torch.mean(tensor, dim=dim, keepdim=True)
            variance = torch.var(tensor, dim=dim, keepdim=True)
            std = torch.sqrt(variance)
            standardized = (tensor - mean) / std
            kurt = torch.mean(standardized**4, dim=dim, keepdim=True) - 3
            return kurt
    
        kurtosis_rank = calculate_kurtosis(rank_ATV_d, dim=-1)
    
    # 计算PATV_d
        PATV_d = mean_rank / std_rank + kurtosis_rank
        PATV_d = torch.where(torch.isinf(PATV_d), torch.tensor(float('nan')), PATV_d)
    
        return PATV_d



class DP_Seed:
#['open','close','high','low']
#['ts_mean(close,20)']
    def __init__(self,D_OHLC,population_size=10):
        self.input=D_OHLC
        self.population_size=population_size
        self.OP_AF2A_func_list=['D_ts_max','D_ts_min',
                                'D_ts_delay','D_ts_delta',
                                'D_ts_detrend','D_ts_std','D_ts_mean'] #你这个class需要用到的算子类别的func_list
        self.OP_A2A_func_list=['D_cs_demean']
        #self.OP_AC2A_func_list=['D_ts_mask_mean','D_ts_mask_std']
    
    def generate_toolbox(self):

        self.pset=gp.PrimitiveSetTyped("MAIN", [TypeA] * len(self.input), TypeA)
        int_values = [int(i) for i in [2, 3, 5, 8, 10 ,30]]   
        #注册需要用到的primitives和terminals
        #AF2A
        for func_name in self.OP_AF2A_func_list:
            func = getattr(OP_AF2A, func_name) 
            self.pset.addPrimitive(func, [TypeA, TypeF], TypeA, name=func_name)
        for constant_value in int_values:
            self.pset.addTerminal(constant_value, int)
        
        #A2A
        func = getattr(OP_A2A, self.OP_A2A_func_list) 
        self.pset.addPrimitive(func, TypeA, TypeA, name=func_name)
       
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("DP_Seed", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=1) #树的深度按需求改
        self.toolbox.register("DP_Seed", tools.initIterate, creator.DP_Seed, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.DP_Seed)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        
    def generate_DP_Seed(self):
        self.individuals_code = self.toolbox.population(n=self.population_size)
        self.individuals_code, self.individuals_str = change_name(self.individuals_code, self.input)

#MP_Seed=['M_ts_mean_left_neighbor(M_O, 5, -1)','M_ts_mean_right_neighbor(M_C, 10, 1)']
#dp_seed=DP_Seed(d_ohlc)
#dp_seed.generate_toolbox()
#dp_seed.generate_MP_Root()

class DV_Seed:
    def __init__(self,D_V,population_size=10):
        self.input=D_V
        self.population_size=population_size
        self.OP_AF2A_func_list=['D_ts_max','D_ts_min',
                                'D_ts_delay','D_ts_delta',
                                'D_ts_mean','D_ts_std','D_ts_detrend'] #你这个class需要用到的算子类别的func_list
        self.OP_A2A_func_list=['D_cs_demean']
        self.OP_AA2A_func_list=['D_at_mean']

    def generate_toolbox(self):
        self.pset=gp.PrimitiveSetTyped("MAIN", [TypeA] * len(self.input), TypeA)
        
        int_values = [int(i) for i in [5, 10, 20, 30, 60]]
        #注册需要用到的primitives和terminals
        for func_name in self.OP_AF2A_func_list:
            func = getattr(OP_AF2A, func_name) 
            self.pset.addPrimitive(func, [TypeA, TypeF], TypeA, name=func_name)
        for constant_value in int_values:
            self.pset.addTerminal(constant_value, int)

        for func_name in self.OP_A2A_func_list:
            func = getattr(OP_A2A, func_name) 
            self.pset.addPrimitive(func, TypeA, TypeA, name=func_name)

        for func_name in self.OP_AA2A_func_list:
            func = getattr(OP_AA2A, func_name) 
            self.pset.addPrimitive(func, [TypeA, TypeA], TypeA, name=func_name)


        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("DV_Seed", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=1) #树的深度按需求改
        self.toolbox.register("DV_Seed", tools.initIterate, creator.DV_Seed, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.DV_Seed)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        
    def generate_DV_Seed(self):
        self.individuals_code = self.toolbox.population(n=self.population_size)
        self.individuals_code, self.individuals_str = change_name(self.individuals_code, self.input)

class MP_Seed:
#['open','close','high','low']
    def __init__(self,M_OHLC,population_size=10):
        self.input=M_OHLC
        self.population_size=population_size
        self.OP_B2A_func_list=['D_Minute_std','D_Minute_mean']

        self.OP_B2B_func_list=['M_cs_demean']
        self.OP_BF2B_func_list=['M_ts_delta','M_ts_mean_left_neighbor',
            'M_ts_mean_mid_neighbor','M_ts_mean_right_neighbor',
            'M_ts_std_left_neighbor','M_ts_std_mid_neighbor',
            'M_ts_std_right_neighbor']


    def generate_toolbox(self):
        self.pset=gp.PrimitiveSetTyped("MAIN", [TypeB] * len(self.input), TypeB)
        int_values = [int(i) for i in [1, 2 ,3 ,5 ,10 , 20, 60]]
        #注册需要用到的primitives和terminals
        for func_name in self.OP_B2A_func_list:
            func = getattr(OP_B2A, func_name) 
            self.pset.addPrimitive(func, TypeB, TypeA, name=func_name)
        
        for constant_value in int_values:
            self.pset.addTerminal(constant_value, int)

        for func_name in self.OP_B2B_func_list:
            func = getattr(OP_B2B, func_name) 
            self.pset.addPrimitive(func, TypeB, TypeB, name=func_name)
        
        for func_name in self.OP_BF2B_func_list:
            func = getattr(OP_BF2B, func_name) 
            self.pset.addPrimitive(func, [TypeB, TypeF], TypeB, name=func_name)
        
       

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("MP_Seed", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=1) #树的深度按需求改
        self.toolbox.register("MP_Seed", tools.initIterate, creator.MP_Seed, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.MP_Seed)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        
    def generate_DV_Seed(self):
        self.individuals_code = self.toolbox.population(n=self.population_size)
        self.individuals_code, self.individuals_str = change_name(self.individuals_code, self.input)

class MV_Seed:

    def __init__(self,M_V,population_size=10):
        self.input=M_V
        self.population_size=population_size


        self.OP_B2B_func_list=['M_cs_demean']
        self.OP_BF2B_func_list=['M_ts_delta','M_ts_mean_left_neighbor',
            'M_ts_mean_mid_neighbor','M_ts_mean_right_neighbor',
            'M_ts_std_left_neighbor','M_ts_std_mid_neighbor',
            'M_ts_std_right_neighbor']

    def generate_toolbox(self):
        self.pset=gp.PrimitiveSetTyped("MAIN", [TypeB] * len(self.input), TypeB)
        
        int_values = [int(i) for i in [1, 2 ,3 ,5 , 10, 20, 60]]
        #注册需要用到的primitives和terminals
        for constant_value in int_values:
            self.pset.addTerminal(constant_value, int)

        for func_name in self.OP_B2B_func_list:
            func = getattr(OP_B2B, func_name) 
            self.pset.addPrimitive(func, TypeB, TypeB, name=func_name)
        
        for func_name in self.OP_BF2B_func_list:
            func = getattr(OP_BF2B, func_name) 
            self.pset.addPrimitive(func, [TypeB, TypeF], TypeB, name=func_name)
        
       

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("MV_Seed", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=1) #树的深度按需求改
        self.toolbox.register("MV_Seed", tools.initIterate, creator.MV_Seed, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.MV_Seed)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        
    def generate_DV_Seed(self):
        self.individuals_code = self.toolbox.population(n=self.population_size)
        self.individuals_code, self.individuals_str = change_name(self.individuals_code, self.input)
