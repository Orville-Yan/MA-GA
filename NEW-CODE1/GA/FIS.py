import sys
sys.path.append("..")

from RPN.RPNbuilder import *

class BK_Algo: #Bron-Kerbosch Algo
    pass

class FactorIntoStorage(RPN_Compiler):
    def __init__(self,new_factor:list):
        super().__init__()
        self.storage_path=None
        self.new_factor=[gp.PrimitiveTree.from_string(RPN, self.pset) for RPN in new_factor]

    def get_exist_factor(self):
        #there we need to set the formal of factor storage
        self.exist_factor=None
        pass

    def greedy_algo(self):
        self.add_factor=[]
        for new_ind in self.new_factor:
            for old_ind in self.exist_factor:
                continue
            continue

    def bk_algo(self):
        pass

    def factor_evaluating(self):
        pass



