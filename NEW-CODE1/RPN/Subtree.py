import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)


from RPN.OrganAbstractClass import *

class SubtreeWithMask(Organ):
    def __init__(self, trunk:list[str], branch:list[str]):
        self.input1 = trunk
        self.input2 = branch
        self.input=self.input1+self.input2
        self.config = globals()[f'{self.__class__.__name__[:7]}_Config']
        self.population_size = self.config.population_size
        self.min_depth, self.max_depth = self.config.depth_range


    def generate_pset(self):
        self.pset=gp.PrimitiveSetTyped('Main',[TypeB]*len(self.input1)+[TypeD]*len(self.input2),TypeA)
        self.pset=self.add_primitive_byfunclist(self.config.OP_BD2A_func_list,'OP_BD2A',self.pset)
        self.pset = self.add_primitive_byfunclist(self.config.OP_BBD2A_func_list, 'OP_BBD2A',self.pset)



class SubtreeNoMask(Organ):
    def __init__(self, trunk):
        self.input = trunk
        self.config = globals()[f'{self.__class__.__name__[:7]}_Config']
        self.population_size = self.config.population_size
        self.min_depth, self.max_depth = self.config.depth_range

    def generate_pset(self):
        self.pset = gp.PrimitiveSetTyped('Main', [TypeB] * len(self.input), TypeA)
        self.pset = self.add_primitive_byfunclist(self.config.OP_B2A_func_list, 'OP_B2A', self.pset)
        self.pset = self.add_primitive_byfunclist(self.config.OP_BB2A_func_list, 'OP_BB2A', self.pset)

if __name__ == "__main__":
    pass

