import sys

sys.path.append('..')

from Tools.GA_tools import *
from OP.ToA import *
from OP.Others import *
from OP.ToB import *

class Seed:
    def __init__(self, input_data, population_size=10):
        self.input=input_data
        self.population_size = population_size
        self.pset = None
        self.toolbox = None

    def generate_toolbox(self):
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        class_name = self.__class__.__name__
        creator.create(class_name, gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=1)
        self.toolbox.register(class_name, tools.initIterate, getattr(creator, class_name), self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, getattr(self.toolbox, class_name))
        self.toolbox.register("compile", gp.compile, pset=self.pset)

    def generate_population(self):
        self.individuals_code = self.toolbox.population(n=self.population_size)
        self.individuals_code, self.individuals_str = change_name(self.individuals_code, self.input)


class DP_Seed(Seed):
    def __init__(self, D_OHLC, population_size=10):
        super().__init__(D_OHLC, population_size)
        self.input = D_OHLC
        self.OP_AF2A_func_list = ['D_ts_max', 'D_ts_min','D_ts_delay', 'D_ts_delta', 'D_ts_mean']

    def add_primitive(self):
        self.pset = gp.PrimitiveSetTyped("MAIN", [TypeA] * len(self.input), TypeA)

        for func_name in self.OP_AF2A_func_list:
            func = getattr(OP_AF2A, func_name)
            self.pset.addPrimitive(func, [TypeA, TypeF], TypeA, name=func_name)

        for constant_value in [2, 3, 5,  10, 20]:
            self.pset.addTerminal(constant_value, TypeF)
        self.pset.addPrimitive(OP_Closure.id_int, TypeF, TypeF, name='id_int')
        self.pset.addPrimitive(OP_Closure.id_tensor, TypeA, TypeA, name='id_tensor')

        super().generate_toolbox()

    def run(self):
        self.add_primitive()
        self.generate_population()


class DV_Seed(Seed):
    def __init__(self, D_V, population_size=10):
        super.__init__(D_V, population_size)
        self.input = D_V
        self.OP_AF2A_func_list = ['D_ts_max', 'D_ts_min',
                                  'D_ts_delay', 'D_ts_delta',
                                  'D_ts_mean']
        self.OP_AA2A_func_list = ['D_at_mean']

    def add_primitive(self):
        self.pset = gp.PrimitiveSetTyped("MAIN", [TypeA] * len(self.input), TypeA)
        int_values = [int(i) for i in [5, 10, 20, 30, 60]]
        for func_name in self.OP_AF2A_func_list:
            func = getattr(OP_AF2A, func_name)
            self.pset.addPrimitive(func, [TypeA, TypeF], TypeA, name=func_name)
        for constant_value in int_values:
            self.pset.addTerminal(constant_value, int)

        for func_name in self.OP_AA2A_func_list:
            func = getattr(OP_AA2A, func_name)
            self.pset.addPrimitive(func, [TypeA, TypeA], TypeA, name=func_name)

        self.pset.addPrimitive(OP_Closure.id_int, int, int, name='id_int')
        self.pset.addPrimitive(OP_Closure.id_tensor, torch.tensor, torch.tensor, name='id_tensor')
        super().generate_toolbox()

    def run(self):
        self.add_primitive()
        self.generate_population()


class MP_Seed(Seed):
    def __init__(self, M_OHLC, population_size=10):
        super().__init__(M_OHLC, population_size)
        self.input = M_OHLC
        self.OP_B2A_func_list = ['D_Minute_std', 'D_Minute_mean']

        self.OP_BF2B_func_list = ['M_ts_delta', 'M_ts_mean_left_neighbor',
                                  'M_ts_mean_mid_neighbor', 'M_ts_mean_right_neighbor']

    def add_primitive(self):
        self.pset = gp.PrimitiveSetTyped("MAIN", [TypeB] * len(self.input), TypeB)
        int_values = [int(i) for i in [1, 2, 3, 5, 10, 20, 60]]
        for func_name in self.OP_B2A_func_list:
            func = getattr(OP_B2A, func_name)
            self.pset.addPrimitive(func, TypeB, TypeA, name=func_name)

        for constant_value in int_values:
            self.pset.addTerminal(constant_value, int)

        for func_name in self.OP_BF2B_func_list:
            func = getattr(OP_BF2B, func_name)
            self.pset.addPrimitive(func, [TypeB, TypeF], TypeB, name=func_name)
        self.pset.addPrimitive(OP_Closure.id_int, int, int, name='id_int')
        self.pset.addPrimitive(OP_Closure.id_tensor, torch.tensor, torch.tensor, name='id_tensor')
        super().generate_toolbox()

    def run(self):
        self.add_primitive()
        self.generate_population()


class MV_Seed(Seed):
    def __init__(self, M_V, population_size=10):
        super().__init__(M_V, population_size)
        self.input = M_V
        self.OP_BF2B_func_list = ['M_ts_delta', 'M_ts_mean_left_neighbor',
                                  'M_ts_mean_mid_neighbor', 'M_ts_mean_right_neighbor']

    def add_primitive(self):
        self.pset = gp.PrimitiveSetTyped("MAIN", [TypeB] * len(self.input), TypeB)
        int_values = [int(i) for i in [1, 2, 3, 5, 10, 20, 60]]
        for constant_value in int_values:
            self.pset.addTerminal(constant_value, int)

        for func_name in self.OP_BF2B_func_list:
            func = getattr(OP_BF2B, func_name)
            self.pset.addPrimitive(func, [TypeB, TypeF], TypeB, name=func_name)
        self.pset.addPrimitive(OP_Closure.id_int, int, int, name='id_int')
        self.pset.addPrimitive(OP_Closure.id_tensor, torch.tensor, torch.tensor, name='id_tensor')
        super().generate_toolbox()

    def run(self):
        self.add_primitive()
        self.generate_population()


