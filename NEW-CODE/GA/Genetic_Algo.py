import sys
sys.path.append('..')

from Tools.DataReader import DataReader
from Tools.BackTest import stratified
from OP import *
from GA import *
import torch

class RPN_Producer:
    def __init__(self, year_list: list[str], population_size=10):
        self.population_size = population_size
        self.year_list = year_list
        self.prepare_data()
    
    def prepare_data(self):
        data_reader = DataReader()
        self.D_O, self.D_H, self.D_L, self.D_C, self.D_V = data_reader.get_Day_data(self.year_list)
        self.M_O, self.M_H, self.M_L, self.M_C, self.M_V = data_reader.get_Minute_data(self.year_list)
        self.D_OHLC = [self.D_O, self.D_H, self.D_L, self.D_C]
        self.M_OHLC = [self.M_O, self.M_H, self.M_L, self.M_C]
        del self.D_O, self.D_H, self.D_L, self.D_C, self.M_O, self.M_H, self.M_L, self.M_C

    def make_seed(self):
        self.dp_seed = DP_Seed(self.D_OHLC, self.population_size)
        self.dv_seed = DV_Seed(self.D_V, self.population_size)
        self.mp_seed = MP_Seed(self.M_OHLC, self.population_size)
        self.mv_seed = MV_Seed(self.M_V, self.population_size)
    
    def make_root(self):
        self.mp_root = MP_Root(self.mp_seed)



class RPN_Compiler:
    def __init__(self,M_OHLCV:[TypeB],D_OHLCV:[TypeA]):
        self.input1=['M_O','M_H','M_L','M_C','M_V']
        self.input2=['D_O','D_H','D_L','D_C','D_V']
        self.data=M_OHLCV+D_OHLCV

        
    def generate_toolbox(self):
        self.pset = gp.PrimitiveSetTyped("MAIN", [TypeB] * len(self.input1)+[TypeA] * len(self.input2), TypeA)
        name=self.input1+self.input2
        for i in range(len(name)):
            self.pset.renameArguments(**{f'ARG{i}': name[i]})
        
        creator.create('Tree_Parser', gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=1)
        self.toolbox.register('Tree_Parser', tools.initIterate, getattr(creator, 'Tree_Parser'), self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, getattr(self.toolbox, 'Tree_Parser'))
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        
    def add_primitive(self):
        pass
    
    def compile(self,deap_formula_str_list):
        individual_list=[]
        deap_formula_code_list=[gp.PrimitiveTree.from_string(k, self.pset) for k in deap_formula_str_list]
        for inidividual in deap_formula_code_list:
            compiled_func = self.toolbox.compile(expr=inidividual)
            individual_list.append(compiled_func(*self.data))
        return individual_list


