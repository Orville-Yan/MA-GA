import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

from OrganAbstractClass import *

class MP_Root(Organ):
    def __init__(self, MP_Seed: list[str]):
        self.input = MP_Seed
        self.config=globals()[f'{self.__class__.__name__[3:]}_Config']
        self.population_size = self.config.population_size
        self.min_depth,self.max_depth=self.config.depth_range

    def generate_pset(self):
        self.pset = gp.PrimitiveSetTyped("MAIN", [TypeB] * len(self.input), TypeB)

        self.pset=self.add_primitive_byfunclist(self.config.OP_B2B_func_list,'OP_B2B',self.pset)
        self.pset = self.add_primitive_byfunclist( self.config.OP_BB2B_func_list,'OP_BB2B' ,self.pset)

class MV_Root(Organ):
    def __init__(self, MV_Seed: list[str]):
        self.input = MV_Seed
        self.config = globals()[f'{self.__class__.__name__[3:]}_Config']
        self.population_size = self.config.population_size
        self.min_depth, self.max_depth = self.config.depth_range

    def generate_pset(self):
        self.pset = gp.PrimitiveSetTyped("MAIN", [TypeB] * len(self.input), TypeB)

        self.pset = self.add_primitive_byfunclist(self.config.OP_B2B_func_list,'OP_B2B', self.pset)
        self.pset = self.add_primitive_byfunclist(self.config.OP_BB2B_func_list,'OP_BB2B', self.pset)

class DP_Root(Organ):
    def __init__(self, DP_Seed: list[str]):
        self.input = DP_Seed
        self.config = globals()[f'{self.__class__.__name__[3:]}_Config']
        self.population_size = self.config.population_size
        self.min_depth, self.max_depth = self.config.depth_range

    def generate_pset(self):
        self.pset = gp.PrimitiveSetTyped("MAIN", [TypeA] * len(self.input), TypeA)

        self.pset = self.add_primitive_byfunclist(self.config.OP_AA2A_func_list,'OP_AA2A', self.pset)
        self.pset = self.add_primitive_byfunclist(self.config.OP_AF2A_func_list,'OP_AF2A', self.pset)
        self.pset = self.add_constant_terminal(self.config.default_lookback,self.pset)


class DV_Root(Organ):
    def __init__(self, DV_Seed: list[str]):
        self.input = DV_Seed
        self.config = globals()[f'{self.__class__.__name__[3:]}_Config']
        self.population_size = self.config.population_size
        self.min_depth, self.max_depth = self.config.depth_range

    def generate_pset(self):
        self.pset = gp.PrimitiveSetTyped("MAIN", [TypeA] * len(self.input), TypeA)

        self.pset = self.add_primitive_byfunclist(self.config.OP_AA2A_func_list,'OP_AA2A', self.pset)
        self.pset = self.add_primitive_byfunclist(self.config.OP_AF2A_func_list,'OP_AF2A', self.pset)
        self.pset = self.add_constant_terminal(self.config.default_lookback,self.pset)





if __name__ == "__main__":
    pass

    

