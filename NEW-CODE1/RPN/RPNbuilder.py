import sys
sys.path.append("..")

from ToolsGA import *
from OP import *
from RPN import *

class config:
    seed_size=10
    root_size=10
    branch_size=10
    trunk_size=10
    subtree_size=10
    tree_size=10


class RPN_Producer:
    def __init__(self):
        self.D_OHLC = ['D_O', 'D_H', 'D_L', 'D_C']
        self.D_V = ['D_V']
        self.M_OHLC = ['M_O', 'M_H', 'M_L', 'M_C']
        self.M_V = ['M_V']

    def generate_seed(self):
        DP_Seed_generator = DP_Seed(self.D_OHLC, config.seed_size)
        DP_Seed_generator.run()
        self.dp_seed = DP_Seed_generator.individuals_str

        DV_Seed_generator = DP_Seed(self.D_V, config.seed_size)
        DV_Seed_generator.run()
        self.dv_seed = DV_Seed_generator.individuals_str

        MV_Seed_generator = MV_Seed(self.M_V, config.seed_size)
        MV_Seed_generator.run()
        self.mv_seed = MV_Seed_generator.individuals_str

        MP_Seed_generator = MP_Seed(self.M_OHLC, config.seed_size)
        MP_Seed_generator.run()
        self.mp_seed = MP_Seed_generator.individuals_str

    def generate_root(self):
        MP_Root_generator = MP_Root(self.mp_seed, config.root_size)
        MP_Root_generator.generate_MP_Root()
        self.mp_root = MP_Root_generator.individuals_str

        DP_Root_generator = DP_Root(self.dp_seed, config.root_size)
        DP_Root_generator.generate_DP_Root()
        self.dp_root = DP_Root_generator.individuals_str

        DV_Root_generator = DV_Root(self.dv_seed, config.root_size)
        DV_Root_generator.generate_DV_Root()
        self.dv_root = DV_Root_generator.individuals_str

        MV_Root_generator = MV_Root(self.mv_seed, config.root_size)
        MV_Root_generator.generate_MV_Root()
        self.mv_root = MV_Root_generator.individuals_str

        self.m_root = self.mv_root + self.mp_root
        self.d_root = self.dv_root + self.dp_root

    def generate_branch(self):
        self.m_branch = []

        M_Branch_MP2D_generator = M_Branch_MP2D(self.mp_seed, config.branch_size)
        M_Branch_MP2D_generator.run()
        self.m_branch.extend(M_Branch_MP2D_generator.individuals_str)

        M_Branch_MPDP2D_generator = M_Branch_MPDP2D(self.mp_seed, self.dp_seed, config.branch_size)
        M_Branch_MPDP2D_generator.run()
        self.m_branch.extend(M_Branch_MP2D_generator.individuals_str)

        M_Branch_MV2D_generator = M_Branch_MV2D(self.mv_seed, config.branch_size)
        M_Branch_MV2D_generator.run()
        self.m_branch.extend(M_Branch_MV2D_generator.individuals_str)

        M_Branch_MVDV2D_generator = M_Branch_MVDV2D(self.mv_seed, self.dv_seed, config.branch_size)
        M_Branch_MVDV2D_generator.run()
        self.m_branch.extend(M_Branch_MVDV2D_generator.individuals_str)

    def generate_trunk(self):
        Trunk_generator = Trunk(self.m_root, self.d_root, self.m_branch, ['industry'], config.trunk_size)
        Trunk_generator.generate_toolbox()
        Trunk_generator.generate_Trunk()
        self.trunk = Trunk_generator.individuals_str

    def generate_subtree(self):
        self.subtree = []

        Subtree_withMask_generator = SubtreeWithMask(self.trunk, self.m_branch, config.subtree_size)
        Subtree_withMask_generator.run()
        self.subtree.extend(Subtree_withMask_generator.individuals_str)

        Subtree_noMask_generator = SubtreeNoMask(self.trunk, config.subtree_size)
        Subtree_noMask_generator.run()
        self.subtree.extend(Subtree_noMask_generator.individuals_str)

    def generate_tree(self):
        Tree_generator = Tree(self.subtree, config.tree_size)
        Tree_generator.generate_toolbox()
        Tree_generator.generate_tree()
        self.tree = Tree_generator.individuals_str

    def run(self):
        self.generate_seed()
        self.generate_root()
        self.generate_branch()
        self.generate_trunk()
        self.generate_subtree()
        self.generate_tree()


class RPN_Compiler:
    def __init__(self):
        self.input1 = ['M_O', 'M_H', 'M_L', 'M_C', 'M_V']
        self.input2 = ['D_O', 'D_H', 'D_L', 'D_C', 'D_V']
        self.input3 = ['industry']
        self.input4 = ['TypeD_data']
        self.input5 = ['TypeC_data']
        self.generate_toolbox()

    def prepare_data(self,year_lst):
        data_reader = ParquetReader()

        Minute_data = data_reader.get_Minute_data(year_lst)

        Day_data = data_reader.get_Day_data(year_lst)

        industry_data = data_reader.get_barra(year_lst)[:, :, 10:]

        TypeD_data = OP_B2D.Mmask_1h_after_open(Minute_data[0])

        TypeC_data = OP_AF2C.Dmask_min(Day_data[0], 20)

        self.data = list(Minute_data) + list(Day_data) + [industry_data] + [TypeD_data] + [TypeC_data]
        del Minute_data, Day_data, industry_data, TypeD_data, TypeC_data

    def generate_toolbox(self):
        self.pset = gp.PrimitiveSetTyped("MAIN", [TypeB] * 5 + [TypeA] * 5 + [TypeE] + [TypeD] + [TypeC], TypeA)
        name = self.input1 + self.input2 + self.input3 + self.input4 + self.input5
        for i in range(len(name)):
            self.pset.renameArguments(**{f'ARG{i}': name[i]})

        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create('Tree_Compiler', gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=10)
        self.toolbox.register('Tree_Compiler', tools.initIterate, getattr(creator, 'Tree_Compiler'), self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, getattr(self.toolbox, 'Tree_Compiler'))
        self.toolbox.register("compile", gp.compile, pset=self.pset)

        self.add_all_primitives()

    def add_primitive_byclass(self, op_classname):
        parts = op_classname.split('2')
        part1 = parts[0].split('_')[1]
        part2 = parts[1]

        op = globals()[op_classname]()
        for func_name in op.func_list:
            func = getattr(op, func_name)
            input_class_list = [globals()[f"Type{char}"] for char in part1]
            output_class = globals()[f"Type{part2}"]
            self.pset.addPrimitive(func, input_class_list, output_class, name=func_name)

    def add_all_primitives(self):
        for type in ['A', 'B', 'C', 'D']:
            class_list = globals()[f"OPclass_name_2{type}"]
            for class_name in class_list:
                self.add_primitive_byclass(class_name)

        self.pset.addPrimitive(OP_Closure.id_int, [TypeF], TypeF, name='id_int')
        self.pset.addPrimitive(OP_Closure.id_industry, [TypeE], TypeE, name='id_industry')
        self.pset.addPrimitive(OP_Closure.id_float, [TypeG], TypeG, name='id_float')

        for constant_value in [int(i) for i in [1, 2, 3, 5, 10, 20, 30, 60]]:
            self.pset.addTerminal(constant_value, TypeF)

        for constant_value in [0.05, 0.1]:
            self.pset.addTerminal(constant_value, TypeG)

    def compile(self, RPN:str):
        inidividual=gp.PrimitiveTree.from_string(RPN, self.pset)
        compiled_func = self.toolbox.compile(expr=inidividual)
        return compiled_func(*self.data)


class RPN_Parser(RPN_Compiler):
    def __init__(self,RPN:str):
        super().__init__()
        self.rpn=RPN
        self.deap_code=gp.PrimitiveTree.from_string(RPN, self.pset)

    def get_tree_structure(self):
        pass

    def plot_tree(self):
        pass

    def parse_from_the_outside_in(self):
        pass

    def parse_from_the_inside_out(self):
        pass

    def parse_tree(self):
        self.tree_A=None
        self.tree_B=None

        self.subtree_A=None
        self.subtree_B=None

        self.trunk_A = None
        self.trunk_B = None

        self.branch_A = []
        self.branch_B = []

        self.root_A = []
        self.root_B = []

        self.seed_A = []
        self.seed_B = []


class RPN_Pruner(RPN_Parser):
    def __init__(self,RPN:str):
        super().__init__(RPN)
        self.get_tree_structure()

    def delete_param_redundancy(self):
        pass

    def delete_closure(self):
        pass

    def prune(self):
        pass
