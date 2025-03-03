import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

from RPN.OrganAbstractClass import *

class M_Branch_MP2D(Organ):
    def __init__(self, mp_root: [str]):
        self.input = mp_root
        self.config = globals()[f'{self.__class__.__name__[2:-5]}_Config']
        self.population_size = self.config.population_size
        self.min_depth, self.max_depth = self.config.depth_range

    def generate_pset(self):
        self.pset = gp.PrimitiveSetTyped("MAIN",[TypeB]*len(self.input), TypeD)
        self.pset=self.add_constant_terminal(self.config.default_lookback,self.pset)
        self.pset = self.add_primitive_byclass('OP_B2D', self.pset)

class M_Branch_MPDP2D(Organ):
    def __init__(self, mp_root: [str], dp_root: [str]):
        self.input_m = mp_root
        self.input_d = dp_root
        self.input=self.input_m+self.input_d
        self.config = globals()[f'{self.__class__.__name__[2:-7]}_Config']
        self.population_size = self.config.population_size
        self.min_depth, self.max_depth = self.config.depth_range

    def generate_pset(self):
        self.pset = gp.PrimitiveSetTyped("MAIN",[TypeB]*len(self.input_m)+[TypeA]*len(self.input_d), TypeD)
        self.pset = self.add_primitive_byclass('OP_BA2D', self.pset)

class M_Branch_MV2D(Organ):
    def __init__(self, mv_root: [str]):
        self.input = mv_root
        self.config = globals()[f'{self.__class__.__name__[2:-5]}_Config']
        self.population_size = self.config.population_size
        self.min_depth, self.max_depth = self.config.depth_range

    def generate_pset(self):
        self.pset = gp.PrimitiveSetTyped("MAIN",[TypeB]*len(self.input), TypeD)
        self.pset = self.add_constant_terminal(self.config.default_lookback,self.pset)
        self.pset = self.add_primitive_byclass('OP_B2D', self.pset)


class M_Branch_MVDV2D(Organ):
    def __init__(self, mv_root: [str], dv_root: [str]):
        self.input_m = mv_root
        self.input_d = dv_root
        self.input=self.input_m+self.input_d
        self.config = globals()[f'{self.__class__.__name__[2:-7]}_Config']
        self.population_size = self.config.population_size
        self.min_depth, self.max_depth = self.config.depth_range

    def generate_pset(self):
        self.pset = gp.PrimitiveSetTyped("MAIN",[TypeB]*len(self.input_m)+[TypeA]*len(self.input_d), TypeD)
        self.pset = self.add_primitive_byclass('OP_BA2D', self.pset)


class D_Branch_DP2C(Organ):
    def __init__(self,dp_root:list[str]):
        self.input=dp_root
        self.config = globals()[f'{self.__class__.__name__[2:-5]}_Config']
        self.population_size = self.config.population_size
        self.min_depth, self.max_depth = self.config.depth_range

    def generate_pset(self):
        self.pset=gp.PrimitiveSetTyped('main',[TypeA]*len(self.input),TypeC)
        self.pset=self.add_primitive_byclass('OP_AF2C',self.pset)
        self.pset=self.add_constant_terminal(self.config.default_lookback,self.pset)


class D_Branch_DV2C(Organ):
    def __init__(self, dv_root: list[str]):
        self.input = dv_root
        self.config = globals()[f'{self.__class__.__name__[2:-5]}_Config']
        self.population_size = self.config.population_size
        self.min_depth, self.max_depth = self.config.depth_range

    def generate_pset(self):
        self.pset = gp.PrimitiveSetTyped('main', [TypeA] * len(self.input), TypeC)
        self.pset = self.add_primitive_byclass('OP_AF2C', self.pset)
        self.pset = self.add_constant_terminal(self.config.default_lookback, self.pset)

if __name__ == "__main__":
    pass
