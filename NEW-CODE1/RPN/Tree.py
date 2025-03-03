import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

from RPN.OrganAbstractClass import *


class Tree(Organ):
    def __init__(self, subtrees: list[str]):#subtree
        self.input = subtrees
        self.config = globals()[f'{self.__class__.__name__}_Config']
        self.population_size = self.config.population_size
        self.min_depth, self.max_depth = self.config.depth_range


    def generate_pset(self):
        self.pset=gp.PrimitiveSetTyped('main',[TypeA]*len(self.input),TypeA)
        self.pset=self.add_primitive_byfunclist(self.config.OP_AF2A_func_list,'OP_AF2A',self.pset)
        self.pset=self.add_constant_terminal(self.config.default_lookback,self.pset)

if __name__ == '__main__':
    pass
