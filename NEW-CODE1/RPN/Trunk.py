import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

from RPN.OrganAbstractClass import *

class Trunk(Organ):
    def __init__(self, M_Root: list[str], D_Root: list[str], TypeC: list[str],TypeD: list[str], ind_str: str):
        self.input1 = M_Root
        self.input2 = D_Root
        self.input3 = TypeD
        self.input4 = TypeC
        self.input5 = [ind_str]
        self.input=self.input1+self.input2+self.input3+self.input4+self.input5
        self.config = globals()[f'{self.__class__.__name__}_Config']
        self.population_size = self.config.population_size
        self.min_depth, self.max_depth = self.config.depth_range

    def generate_pset(self):
        self.pset=gp.PrimitiveSetTyped("MAIN",
                                       [TypeB] * len(self.input1)+[TypeA]*len(self.input2)+[TypeD]*len(self.input3)+[TypeC]*len(self.input4)+[TypeE],
                                       TypeB)
        for type in ['A','B','C','D']:
            class_list=globals()[f'OPclass_name_2{type}']
            for class_name in class_list:
                self.pset=self.add_primitive_byclass(class_name,self.pset)

        self.pset=self.add_constant_terminal(self.config.default_lookback,self.pset)
        self.pset=self.add_float_terminal(self.config.default_float,self.pset)
        self.pset.addPrimitive(OP_Closure.id_industry, [TypeE], TypeE, 'id_industry')



if __name__ == '__main__':
    pass
