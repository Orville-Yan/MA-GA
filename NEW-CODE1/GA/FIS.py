import sys
import os
import networkx as nx
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

from deap import creator, base, tools, gp
import operator
import random
from deap import gp
import networkx as nx
from itertools import combinations
from RPN.RPNbuilder import *
import difflib
# 调了一个用了算字符串相似度的包difflib用来算边
class BK_Algo: #Bron-Kerbosch Algo
    # 直接调用networkx库内置的最大团算法？也可以调出源码放在这里。
    pass


class FactorIntoStorage(RPN_Compiler):
    def __init__(self,new_factor:list):
        super().__init__()
        self.storage_path=None
        self.new_factor=[gp.PrimitiveTree.from_string(RPN, self.pset) for RPN in new_factor]

    def get_exist_factor(self):
        existing_factors = []
        if self.storage_path:  
            with open(self.storage_path, 'r', encoding='utf-8') as file:
                for line in file:
                    existing_factors.append(gp.PrimitiveTree.from_string(line.strip(), self.pset))
        self.exist_factor = existing_factors 

    def greedy_algo(self):
        selected_factors = []
        for new_factor in self.new_factor:
            should_add = True
            for exist_factor in self.exist_factor:
                similarity = self.calculate_similarity(new_factor, exist_factor)
                if similarity >= 0.9:  
                    should_add = False
                    break
            if should_add:
                selected_factors.append(new_factor)
        self.greedy_factor = selected_factors
    @staticmethod
   
    def calculate_similarity(factor1, factor2):
        matcher = difflib.SequenceMatcher(None,str(factor1),str(factor2))
        return  matcher.ratio()
    def build_factor_graph(self,old_strings:str, new_strings:str):
        graph = nx.Graph()
        all_strings = old_strings + new_strings
        graph.add_nodes_from(all_strings)
        for factor1, factor2 in combinations(all_strings, 2):
            similarity = self.calculate_similarity(factor1, factor2)
            if similarity >= 0.6: 
                graph.add_edge(factor1, factor2, weight=similarity)
        return graph

    def bk_algo(self):
        old_strings = [str(factor) for factor in self.exist_factor]
        new_strings = [str(factor) for factor in self.new_factor]
        graph = self.build_factor_graph(old_strings, new_strings)
        # nx内置的极大团算法，有源码可以添加
        maximal_cliques = list(nx.find_cliques(graph))
        best_clique = max(maximal_cliques, key=len)
        selected_strings = [factor for factor in best_clique if factor in new_strings]
        self.bk_factor = selected_strings
    def factor_evaluating(self):
        pass

    def store_factors(self):
        self.add_factor = self.bk_factor
        if self.storage_path:
            with open(self.storage_path, 'a', encoding='utf-8') as file:
                for factor in self.add_factor:
                    file.write(factor + '\n')
        else:
            raise ValueError("存储路径未定义")
if __name__ == "__main__":
    from deap import creator,base,tools
    pset = gp.PrimitiveSet("MAIN", arity=2) 
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)  
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create('Individual',gp.PrimitiveTree,fitness = creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register('expr',gp.genHalfAndHalf,pset=pset, min_=4, max_=4)
    toolbox.register('Individual',tools.initIterate,creator.Individual,toolbox.expr)
    toolbox.register('population',tools.initRepeat,list,toolbox.Individual)
    code = toolbox.population(20)
    FIS = FactorIntoStorage([''])
    FIS.new_factor = code[:10]
    FIS.exist_factor = code[10:]
    FIS.bk_algo()
    FIS.greedy_algo()
    print(FIS.bk_factor)
