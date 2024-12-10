import sys
sys.path.append('..')

from Tools.DataReader import DataReader
from Tools.BackTest import stratified
from OP import *
from GA import *
import torch

class GeneticAlgorithm:
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