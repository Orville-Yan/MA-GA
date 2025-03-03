from OPT import *
from FIS import *
class config:
    warm_start_time=[2016,2017]

    warm_start_period=50

    warm_start_threshold=0.8

    bins_num=5

    freque=5

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

            fitness=backtest.get_long_over_mean_sharpe()

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
