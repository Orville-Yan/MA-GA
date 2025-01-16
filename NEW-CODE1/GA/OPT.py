from RPN import *
from ToolsGA import *
class MAP:
    def __init__(self,SearchSpace:[[str]]):

        self.SearchSpace=SearchSpace

    def space_map(self):

        self.PathSpace=None

    def vector_map(self,path_vector):

        search_vector=None

        return search_vector

    def vector_compile(self,vector):

        rpn=''

        return rpn


class GA_optimizer(MAP):
    def __init__(self,SearchSpace:[int],time_seq,bins_num):
        super().__init__(SearchSpace)
        self.pathspace=self.space_map()
        self.good_individuals=[[int]]

        self.rpn_compiler=RPN_Compiler()
        self.rpn_compiler.prepare_data(time_seq)

        reader=ParquetReader()
        self.labels=reader.get_labels(time_seq)

        self.bins_num=bins_num
        self.period_num=250/self.bins_num


    def initial_group(self):
        pass

    def cross_variation(self):
        pass

    def point_variation(self):
        pass

    def itreation(self):
        pass


    def fitness(self,ind):

        search_vector=self.vector_map(ind)

        rpn=self.vector_compile(search_vector)

        factor=self.rpn_compiler.compile(rpn)

        backtest = FactorTest(factor,self.labels,self.bins_num)

        fitness=backtest.get_long_over_mean_sharpe(self.period_num)

        return fitness

    def run(self):
        pass
