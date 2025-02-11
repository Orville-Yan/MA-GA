class config:
    warm_start_time=[2016,2017]

    warm_start_period=50

    warm_start_threshold=0.8

    similarity_threhold = 0.6

    default_population = 10

    bins_num=5

    freque=5
    
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)
from .OPT import *
from .FIS import *
from RPN.RPNbuilder import RPN_Compiler

class GroupTest(RPN_Compiler):
    def __init__(self,factor_list:[str]):
        super().__init__()
        self.factors=factor_list
        self.in_sample_time=range(2010,2020)
        self.out_sample_time=range(2019,2022)

    def in_sample_response_rate(self):
        ft = FactorTest(factor_tensor, factor_target, self.in_sample_time, bins_num=5, factor_name='factor')
        ft.plot(output_pdf='in_sample_output.pdf')
        
    def out_sample_response_rate(self):
        ft = FactorTest(factor_tensor, factor_target, self.out_sample_time, bins_num=5, factor_name='factor')
        ft.plot(output_pdf='out_sample_output.pdf')



class GA:
    def __init__(self):
        pass

    def get_originals(self):

        rpn_producer=RPN_Producer()

        rpn_producer.run()

        self.orignals=rpn_producer.tree

    def warm_start(self):
        reader=ParquetReader()

        factor_target=reader.get_labels(config.warm_start_time,config.freque)

        originals={}

        for ind in self.orignals:

            rpn_compiler=RPN_Compiler()

            rpn_compiler.prepare_data(config.warm_start_time)

            factor=rpn_compiler.compile(ind)

            backtest=FactorTest(factor,factor_target,config.bins_num)

            fitness=backtest.get_long_over_mean_sharpe(config.warm_start_period)

            if fitness >= config.warm_start_threshold:

                originals[ind]=fitness

        self.orignals=originals

    def get_ancestors(self):

        self.ancestors={str:float}

    def get_ProdSpace(self):

        self.search_space={str:[[str]]}

    def GA_optimize(self):

        optimizer=GA_optimizer(self.search_space,config.warm_start_time,config.bins_num)

        optimizer.run()

        self.good_individuals=optimizer.good_individuals

    def maximum_clique(self):

        fis=FactorIntoStorage(self.good_individuals)

        fis.bk_algo()

        self.maximum_clique=[]
