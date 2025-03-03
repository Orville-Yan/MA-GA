import sys
import os
import networkx as nx

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

from deap import gp
import networkx as nx
import pandas as pd
from itertools import combinations
from RPN.RPNbuilder import *
import difflib
from GA.Config import FIS_Config as Config

class BK_Algo:  # Bron-Kerbosch Algo
    pass


class FactorIntoStorage(RPN_Compiler):
    def __init__(self, new_factor: list,year_lst:list):
        super().__init__(year_lst)
        self.storage_path = Config.storage_path
        self.new_factor = new_factor

    def get_exist_factor(self):
        if os.path.exists(self.storage_path):
            self.factor_storage = pd.read_excel(self.storage_path)
            self.exist_factor = list(self.factor_storage['tree'])
        else:
            self.factor_storage = pd.DataFrame(columns = ['tree','in_sample','out_sample','annual_yield','trunk','root','seed','branch','subtree'])
            self.exist_factor = []

    def greedy_algo(self):
        selected_factors = []
        exist_factor = self.exist_factor
        for new_factor in self.new_factor:
            should_add = True
            for factor in exist_factor:
                similarity = self.calculate_similarity(new_factor, factor)
                if similarity >= Config.similarity_threhold:
                    should_add = False
                    break
            if should_add:
                selected_factors.append(new_factor)
                exist_factor.append(new_factor)
        self.greedy_factor = selected_factors

    @staticmethod
    def calculate_similarity(factor1: gp.PrimitiveTree, factor2: gp.PrimitiveTree):
        matcher = difflib.SequenceMatcher(None, factor1, factor2)
        return matcher.ratio()

    def build_factor_graph(self, factors_lst: list):
        graph = nx.Graph()
        graph.add_nodes_from(factors_lst)
        for factor1, factor2 in combinations(factors_lst, 2):
            similarity = self.calculate_similarity(factor1, factor2)
            if similarity < Config.similarity_threhold:
                graph.add_edge(factor1, factor2, weight=similarity)
        return graph

    def bk_algo(self):
        new_factors = self.new_factor
        graph = self.build_factor_graph(new_factors)
        maximal_cliques = list(nx.find_cliques(graph))
        best_clique = max(maximal_cliques, key=len)
        selected_strings = [factor for factor in best_clique if factor in new_factors]
        self.bk_factor = selected_strings

    def factor_evaluating(self):
        pass

    def store_factors(self):
        self.add_factor = self.bk_factor
        factor_storage = self.factor_storage

        def tree2lst(tree):
            if type(tree) == list:
                return [parser.tree2str(x) for x in tree]
            else:
                return parser.tree2str(tree)

        for factor in self.add_factor:
            parser = RPN_Parser(factor)
            parser.get_tree_structure()
            parser.parse_tree()
            tree = parser.tree2str(parser.tree['tree_mode'])
            trunks = tree2lst(parser.trunk['tree_mode'])
            roots = tree2lst(parser.root['tree_mode'])
            seeds = tree2lst(parser.seed['tree_mode'])
            branches = tree2lst(parser.branch['tree_mode'])
            subtrees = tree2lst(parser.subtree['tree_mode'])
            factor_storage.loc[factor_storage.shape[0] + 1] = {'tree': tree, 'trunk': trunks, 'root': roots,
                                                               'seed': seeds, 'branch': branches, 'subtree': subtrees}
        factor_storage.to_excel(self.storage_path)


if __name__ == "__main__":
    producer = RPN_Producer()
    producer.run()
    FIS = FactorIntoStorage(producer.tree,[2016])
    FIS.get_exist_factor()
    FIS.bk_algo()
    # FIS.store_factors()

