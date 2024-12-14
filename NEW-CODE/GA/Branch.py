import sys
sys.path.append('..')

from OP import *
from GA.Root import *
from ToolsGA.GA_tools import *
import torch
from deap import gp, base, tools, creator


class Branch:
    def __init__(self, mp_root: MP_Root, population_size: int):
        self.population_size = population_size
        self.pset = None
        self.toolbox = None
        self.int_values = [torch.tensor(i, dtype=torch.int) for i in [2, 3, 5, 8, 10, 30]]

    def generate_toolbox(self):
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        class_name = self.__class__.__name__
        creator.create(class_name, gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=1)
        self.toolbox.register(class_name, tools.initIterate, getattr(creator,class_name), self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, getattr(self.toolbox,class_name))
        self.toolbox.register("compile", gp.compile, pset=self.pset)

    def generate_population(self):
        self.individuals_code = self.toolbox.population(n=self.population_size)
        self.individuals_code, self.individuals_str = change_name(self.individuals_code, self.input)


class M_Branch_MP2D(Branch):
    def __init__(self, mp_root: MP_Root, population_size=10):
        super().__init__(mp_root, population_size)
        self.input = mp_root

    def add_primitive(self):
        self.pset = gp.PrimitiveSetTyped("MAIN",[TypeB]*len(self.input), TypeD)   
            
        # 添加TypeF
        for constant in self.int_values:
            self.pset.addTerminal(constant,TypeF)
        self.pset.addPrimitive(OP_Closure.id_int,[TypeF],TypeF)
        # 添加primitive
        op = OP_BF2D()
        for name in op.func_list:
            func = getattr(op, name)
            self.pset.addPrimitive(func, [TypeB,TypeF], TypeD)
        op = OP_B2D()
        for name in op.func_list:
            func = getattr(op, name)
            self.pset.addPrimitive(func, [TypeB], TypeD)
        # 创建工具箱
        super().generate_toolbox()

    def run(self):
        self.add_primitive()
        self.generate_population()

class M_Branch_MPDP2D(Branch):
    # missing DP_Root
    def __init__(self, mp_root: MP_Root, dp_root: DP_Root,population_size=10):
        # 暂时没有DP
        super().__init__(mp_root, population_size)
        self.input_m = mp_root
        self.input_d = dp_root
        self.input = self.input_m + self.input_d

    def add_primitive(self):
        self.pset = gp.PrimitiveSetTyped("MAIN",[TypeB]*len(self.input_m)+[TypeA]*len(self.input_d), TypeD)
        # 添加primitive
        op = OP_BA2D()
        for name in op.func_list:
            func = getattr(op, name)
            self.pset.addPrimitive(func, [TypeB,TypeA], TypeD)
        # 创建工具箱
        super().generate_toolbox()

    def run(self):
        self.add_primitive()
        self.generate_population()

class M_Branch_MV2D(Branch):
    # missing MV_Root
    def __init__(self, mv_root: MV_Root, population_size=10):
        super().__init__(mv_root, population_size)
        self.input = mv_root
    def add_primitive(self):
        self.pset = gp.PrimitiveSetTyped("MAIN",[TypeB]*len(self.input), TypeD)
        # 添加TypeF
        for constant in self.int_values:
            self.pset.addTerminal(constant,TypeF)
        self.pset.addPrimitive(OP_Closure.id_int,[TypeF],TypeF)
        # 添加primitive
        op = OP_BF2D()
        for name in op.func_list:
            func = getattr(op, name)
            self.pset.addPrimitive(func, [TypeB,TypeF], TypeD)
        op = OP_B2D()
        for name in op.func_list:
            func = getattr(op, name)
            self.pset.addPrimitive(func, [TypeB], TypeD)
        # 创建工具箱
        super().generate_toolbox()

    def run(self):
        self.add_primitive()
        self.generate_population()

class M_Branch_MVDV2D(Branch):
    # 暂时没有DV
    def __init__(self, mv_root: MV_Root, dv_root: DV_Root,population_size=10):
        super().__init__(mv_root, population_size)
        self.input_m = mv_root
        self.input_d = dv_root
        self.input = self.input_m + self.input_d

    def add_primitive(self):
        self.pset = gp.PrimitiveSetTyped("MAIN",[TypeB]*len(self.input_m)+[TypeA]*len(self.input_d), TypeD)
   
        # 添加primitive
        op = OP_BA2D()
        for name in op.func_list:
            func = getattr(op, name)
            self.pset.addPrimitive(func, [TypeB,TypeA], TypeD)
        # 创建工具箱
        super().generate_toolbox()

    def run(self):
        self.add_primitive()
        self.generate_population()

if __name__ == "__main__":
    mp_root = ['M_cs_scale(M_ts_mean_left_neighbor(M_O, 5, -1))', 'M_ts_pctchg(M_ts_mean_right_neighbor(M_C, 10, 1))', 'M_cs_zscore(M_ts_mean_left_neighbor(M_O, 5, -1))', 'M_cs_rank(M_ts_mean_left_neighbor(M_O, 5, -1))']
    dp_root = ['D_ts_pctchg(D_at_abs(D_O))','D_ts_pctchg(D_ts_max(D_O,5))']
    m_branch_mp2d = M_Branch_MP2D(mp_root)
    m_branch_mp2d.run()
    print("M_Branch_MP2D Individual Str: ", m_branch_mp2d.individuals_str)
    print("M_Branch_MP2D Individual Code: ", m_branch_mp2d.individuals_code)

    m_branch_mpdp2d = M_Branch_MPDP2D(mp_root,dp_root)

    m_branch_mpdp2d.generate_toolbox()
    print(type(m_branch_mpdp2d.population_size))
    m_branch_mpdp2d.run()

    print("M_Branch_MP2D Individual Str: ", m_branch_mpdp2d.individuals_str)
    print("M_Branch_MP2D Individual Code: ", m_branch_mpdp2d.individuals_code)
