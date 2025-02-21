from abc import ABC, abstractmethod
from ToolsGA.GA_tools import *
from OP import *

default_lookback= [2, 3, 5, 10, 20]

class Seed_Config:
    population_size = 10
    depth_range=[1,1]
    default_lookback = default_lookback

    OP_AF2A_func_list = ['D_ts_max', 'D_ts_min','D_ts_delay', 'D_ts_delta', 'D_ts_mean']
    OP_AA2A_func_list = ['D_at_mean']
    OP_BF2B_func_list = ['M_ts_delay', 'M_ts_mean_left_neighbor',
                                  'M_ts_mean_mid_neighbor', 'M_ts_mean_right_neighbor']
    OP_BB2B_func_list = []

class Root_Config:
    population_size = 10
    depth_range=[1,1]
    default_lookback = default_lookback

    OP_B2B_func_list = ['M_cs_rank', 'M_cs_scale', 'M_cs_zscore']
    OP_BB2B_func_list = ['M_at_div']

    OP_AA2A_func_list = ['D_at_div']
    OP_AF2A_func_list = ['D_ts_pctchg','D_ts_norm']

class Branch_Config:
    population_size = 10
    depth_range=[1,1]
    default_lookback = default_lookback

class Trunk_Config:
    population_size = 10
    depth_range = [5, 8]
    default_lookback = default_lookback
    default_float = [0.05, 0.1]

class Subtree_Config:
    population_size = 10
    depth_range=[1,1]

    OP_B2A_func_list = ['D_Minute_std', 'D_Minute_mean', 'D_Minute_trend']
    OP_BB2A_func_list  = ['D_Minute_corr', 'D_Minute_weight_mean']
    OP_BD2A_func_list = ['D_Minute_area_mean', 'D_Minute_area_std', 'D_Minute_area_sum', 'D_Minute_area_prod']
    OP_BBD2A_func_list = ['D_Minute_area_weight_mean', 'D_Minute_area_corr', 'D_Minute_area_rankcorr',
                                   'D_Minute_area_bifurcate_mean', 'D_Minute_area_bifurcate_std']

class Tree_Config:
    population_size = 10
    depth_range=[1,1]
    default_lookback = [2, 3, 5, 10, 20]

    OP_AF2A_func_list = ['D_ts_mean', 'D_ts_harmonic_mean', 'D_ts_std']


class Organ(ABC):
    @abstractmethod
    def generate_pset(self):
        pass

    def add_primitive_byclass(self, op_classname, pset):
        parts = op_classname.split('2')
        part1 = parts[0].split('_')[1]
        part2 = parts[1]

        op = globals()[op_classname]()
        for func_name in op.func_list:
            func = getattr(op, func_name)
            input_class_list = [globals()[f"Type{char}"] for char in part1]
            output_class = globals()[f"Type{part2}"]
            pset.addPrimitive(func, input_class_list, output_class, name=func_name)
        return pset

    def add_primitive_byfunclist(self,func_list:list[str],op_classname:str,pset):
        parts = op_classname.split('2')
        part1 = parts[0].split('_')[1]
        part2 = parts[1]

        for func_name in func_list:
            func=getattr(globals()[op_classname],func_name)
            input_class_list = [globals()[f"Type{char}"] for char in part1]
            output_class = globals()[f"Type{part2}"]
            pset.addPrimitive(func, input_class_list, output_class, name=func_name)

        return pset

    def add_constant_terminal(self,constants:list[int],pset):
        for constant_value in constants:
            pset.addTerminal(constant_value,TypeF)
        pset.addPrimitive(OP_Closure.id_int, [TypeF], TypeF, name='id_int')
        return pset

    def add_float_terminal(self,floats:list[float],pset):
        for float_value in floats:
            pset.addTerminal(float_value,TypeG)
        pset.addPrimitive(OP_Closure.id_float, [TypeG], TypeG, name='id_float')
        return pset

    def generate_toolbox(self,classname,pset,max_depth=1,min_depth=1):
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create(classname,gp.PrimitiveTree,pset = pset)
        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_= min_depth, max_=max_depth)
        toolbox.register(classname, tools.initIterate, getattr(creator, classname), toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, getattr(toolbox, classname))
        return toolbox

    def generate_population(self,toolbox,population_size,input_name):
        individuals_code = toolbox.population(n=population_size)
        individuals_code, individuals_str = change_name(individuals_code, input_name)
        return individuals_code, individuals_str

    def generator(self,generator):
        generator.generate_pset()
        toolbox=self.generate_toolbox(classname=generator.__class__.__name__,
                                      pset=generator.pset,
                                      max_depth=generator.max_depth,
                                      min_depth=generator.min_depth)
        organ = self.generate_population(toolbox=toolbox,
                                         population_size=generator.population_size,
                                         input_name=generator.input)[1]
        return organ

class General_pset(Organ):
    def __init__(self):
        self.input1 = ['M_O', 'M_H', 'M_L', 'M_C', 'M_V']
        self.input2 = ['D_O', 'D_H', 'D_L', 'D_C', 'D_V']
        self.input3 = ['industry']
        self.input4 = ['TypeD_data']
        self.input5 = ['TypeC_data']
        self.generate_pset()

    def generate_pset(self):
        self.pset = gp.PrimitiveSetTyped("MAIN", [TypeB] * 5 + [TypeA] * 5 + [TypeE] + [TypeD] + [TypeC], TypeA)
        name = self.input1 + self.input2 + self.input3 + self.input4 + self.input5
        for i in range(len(name)):
            self.pset.renameArguments(**{f'ARG{i}': name[i]})

        self.add_all_primitives('pset')

    def add_all_primitives(self, pset_name):
        pset = getattr(self, pset_name)

        for type in ['A', 'B', 'C', 'D']:
            class_list = globals()[f"OPclass_name_2{type}"]
            for class_name in class_list:
                self.add_primitive_byclass(class_name, pset)

        pset.addPrimitive(OP_Closure.id_int, [TypeF], TypeF, name='id_int')
        pset.addPrimitive(OP_Closure.id_industry, [TypeE], TypeE, name='id_industry')
        pset.addPrimitive(OP_Closure.id_float, [TypeG], TypeG, name='id_float')
        pset.addPrimitive(OP_Closure.id_tensor, [TypeA], TypeA, name='id_tensor')

        for constant_value in [int(i) for i in [1, 2, 3, 5, 10, 20, 30, 60]]:
            pset.addTerminal(constant_value, TypeF)

        for constant_value in [0.05, 0.1]:
            pset.addTerminal(constant_value, TypeG)

general_pset=General_pset()