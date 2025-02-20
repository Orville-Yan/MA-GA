import sys
import os
import networkx as nx

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)


from ToolsGA import *
from OP import *
from RPN import *
import torch
from tqdm import tqdm
from GA.Config import RPNbuilder_Config as config




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

        DV_Seed_generator = DV_Seed(self.D_V, config.seed_size)
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
        MP_Root_generator.run()
        self.mp_root = MP_Root_generator.individuals_str

        DP_Root_generator = DP_Root(self.dp_seed, config.root_size)
        DP_Root_generator.run()
        self.dp_root = DP_Root_generator.individuals_str

        DV_Root_generator = DV_Root(self.dv_seed, config.root_size)
        DV_Root_generator.run()
        self.dv_root = DV_Root_generator.individuals_str

        MV_Root_generator = MV_Root(self.mv_seed, config.root_size)
        MV_Root_generator.run()
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
        Trunk_generator.run() 
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
        Tree_generator.run()
        self.tree = Tree_generator.individuals_str

    def run(self):
        self.generate_seed()
        self.generate_root()
        self.generate_branch()
        self.generate_trunk()
        self.generate_subtree()
        self.generate_tree()


class General_toolbox:
    def __init__(self):
        self.input1 = ['M_O', 'M_H', 'M_L', 'M_C', 'M_V']
        self.input2 = ['D_O', 'D_H', 'D_L', 'D_C', 'D_V']
        self.input3 = ['industry']
        self.input4 = ['TypeD_data']
        self.input5 = ['TypeC_data']
        self.generate_ini_toolbox()

    def generate_ini_toolbox(self):
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

        self.add_all_primitives('pset')

    def add_primitive_byclass(self, op_classname, pset_name):
        parts = op_classname.split('2')
        part1 = parts[0].split('_')[1]
        part2 = parts[1]

        op = globals()[op_classname]()
        for func_name in op.func_list:
            func = getattr(op, func_name)
            input_class_list = [globals()[f"Type{char}"] for char in part1]
            output_class = globals()[f"Type{part2}"]
            pset = getattr(self, pset_name)
            pset.addPrimitive(func, input_class_list, output_class, name=func_name)

    def add_all_primitives(self, pset_name):
        pset = getattr(self, pset_name)

        for type in ['A', 'B', 'C', 'D']:
            class_list = globals()[f"OPclass_name_2{type}"]
            for class_name in class_list:
                self.add_primitive_byclass(class_name, pset_name)

        pset.addPrimitive(OP_Closure.id_int, [TypeF], TypeF, name='id_int')
        pset.addPrimitive(OP_Closure.id_industry, [TypeE], TypeE, name='id_industry')
        pset.addPrimitive(OP_Closure.id_float, [TypeG], TypeG, name='id_float')
        pset.addPrimitive(OP_Closure.id_tensor, [TypeA], TypeA, name='id_tensor')

        for constant_value in [int(i) for i in [1, 2, 3, 5, 10, 20, 30, 60]]:
            pset.addTerminal(constant_value, TypeF)

        for constant_value in [0.05, 0.1]:
            pset.addTerminal(constant_value, TypeG)


class RPN_Parser:
    def __init__(self, RPN, pset):

        self.rpn = RPN
        self.pset = pset
        self.deap_code = gp.PrimitiveTree.from_string(RPN, pset)
        self.tree_structure = None
        self.tree = None

    def get_abbrnsub(self, ac_tree, substr, flag=0, count=0):
        flag = max(flag - 1, 0)
        abbr = ac_tree.root_node.name + '('
        sub = []
        for i, node in enumerate(ac_tree.nodes):
            if i > 0:
                abbr += ', '
            if isinstance(node, Acyclic_Tree):
                if flag == 0:
                    abbr += f'{substr}_ARG{count}'
                    count += 1
                    sub.append(node)
                else:
                    sub_abbr, sub_sub, count = self.get_abbrnsub(node, substr, flag, count)
                    abbr += sub_abbr
                    sub.extend(sub_sub)

            elif isinstance(node, gp.Terminal):
                abbr += str(node.value)
            else:
                raise ValueError('instance error')
        return abbr + ')', sub, count

    def get_tree_depth(self, tree):
        if not isinstance(tree, Acyclic_Tree):
            raise ValueError("instance error")
        max_depth = 0
        for node in tree.nodes:
            if isinstance(node, Acyclic_Tree):
                node_depth = self.get_tree_depth(node)
            else:
                node_depth = 0
            max_depth = max(max_depth, node_depth)
        return max_depth + 1

    def tree2str(self, tree):
        depth = self.get_tree_depth(tree)
        abbr, _, _ = self.get_abbrnsub(tree, '', depth)
        return abbr

    def get_tree_structure(self):
        self.tree_structure = Acyclic_Tree(self.rpn, self.pset)
        return self.tree_structure

    def plot_tree(self, node=None, level=0):
        if self.tree_structure is None:
            self.get_tree_structure()

        if node is None:
            node = self.tree_structure

        # 打印当前节点
        indent = "    " * level
        node_label = node.root_node.name if isinstance(node.root_node, gp.Primitive) else node.root_node.value
        print(f"{indent}{node_label}")

        # 递归打印子节点
        for child in node.nodes:
            if isinstance(child, Acyclic_Tree):
                self.plot_tree(child, level + 1)
            else:
                print(f"    " * (level + 1) + str(child.value))

    def parse_tree(self):
        abbr_tree, sub_tree, _ = self.get_abbrnsub(self.tree_structure, 'subtree')
        self.tree = {
            'abbreviation': abbr_tree,
            'tree_mode': self.tree_structure,
        }

        abbr_subtree = ''
        branch_lst = []
        trunk_lst = []
        for tree in sub_tree:
            count_branch = 0
            count_trunk = 0
            abbr_subtree += tree.root_node.name + '('
            for i, node in enumerate(tree.nodes):
                if i > 0:
                    abbr_subtree += ', '
                if isinstance(node, Acyclic_Tree):
                    if i < len(tree.nodes) - 1 or 'TypeD' not in [t.__name__ for t in tree.root_node.args]:
                        abbr_subtree += f'trunk_ARG{count_trunk}'
                        count_trunk += 1
                        trunk_lst.append(node)
                    else:
                        abbr_subtree += f'branch_ARG{count_branch}'
                        count_branch += 1
                        branch_lst.append(node)

                elif isinstance(node, gp.Terminal):
                    abbr_subtree += str(node.value)
                else:
                    raise ValueError('instance error')
            abbr_subtree += ')'

        self.subtree = {
            'abbreviation': abbr_subtree,
            'tree_mode': sub_tree[0],
        }

        abbr_trunk = []
        sub_trunk = []
        count = 0
        for trunk in trunk_lst:
            trunk_depth = self.get_tree_depth(trunk) - 2
            abbr, sub, count = self.get_abbrnsub(trunk, 'root', trunk_depth, count)
            abbr_trunk.append(abbr)
            sub_trunk.extend(sub)

        self.trunk = {
            'abbreviation': abbr_trunk,
            'tree_mode': trunk_lst,
        }
        count = 0
        abbr_root = []
        sub_root = []
        for root in sub_trunk:
            abbr, sub, count = self.get_abbrnsub(root, 'seed', 0, count)
            abbr_root.append(abbr)
            sub_root.extend(sub)

        abbr_branch = []
        for branch in branch_lst:
            branch_depth = self.get_tree_depth(branch) - 1
            abbr, sub, count = self.get_abbrnsub(branch, 'seed', branch_depth, count)
            abbr_branch.append(abbr)
            sub_root.extend(sub)

        self.branch = {
            'abbreviation': abbr_branch,
            'tree_mode': branch_lst,
        }

        self.root = {
            'abbreviation': abbr_root,
            'tree_mode': sub_trunk,
        }

        self.seed = {
            'abbreviation': [seed.abbreviation for seed in sub_root],
            'tree_mode': sub_root,
        }

    def argnorm(self, seed_str):
        map = {
            'ARG0': 'D_O',
            'ARG1': 'D_C',
            'ARG2': 'D_H',
            'ARG3': 'D_L',
            'ARG4': 'D_V',
            'ARG5': 'M_O',
            'ARG6': 'M_C',
            'ARG7': 'M_H',
            'ARG8': 'M_L',
            'ARG9': 'M_V'
        }
        for key, value in map.items():
            seed_str = seed_str.replace(value, key)
        return seed_str

    def tree2dict(self):
        if not self.tree:
            self.parse_tree()
        self.tree_dict = {
            'tree': [self.tree['abbreviation']],
            'subtree': [self.subtree['abbreviation']],
            'branch': self.branch['abbreviation'],
            'trunk': self.trunk['abbreviation'],
            'root': self.root['abbreviation'],
            'seed': [self.argnorm(seed) for seed in self.seed['abbreviation']]
        }
        return self.tree_dict


class RPN_Compiler:
    def __init__(self, year_list, device=torch.device("cuda")):
        self.general_toolbox=General_toolbox()
        self.general_pset=self.general_toolbox.pset
        self.device = device
        self.prepare_data(year_list)


    def prepare_data(self, year_lst):
        self.data_reader = MmapReader()
        self.Day_data = list(self.data_reader.get_Day_data(year_lst))
        self.Day_data=[i.to(self.device) for i in self.Day_data]
        self.day_list = self.data_reader.get_daylist(year_lst)

        self.industry_data = [self.data_reader.get_Barra(year_lst)[:, :, 10:].to(self.device)]
        self.TypeC_data = [OP_AF2C.Dmask_min(self.Day_data[0], 20)]

    def add_primitive_byclass(self, op_classname, pset_name):
        parts = op_classname.split('2')
        part1 = parts[0].split('_')[1]
        part2 = parts[1]

        op = globals()[op_classname]()
        for func_name in op.func_list:
            func = getattr(op, func_name)
            input_class_list = [globals()[f"Type{char}"] for char in part1]
            output_class = globals()[f"Type{part2}"]
            pset = getattr(self, pset_name)
            pset.addPrimitive(func, input_class_list, output_class, name=func_name)

    def split_segment(self, rpn):
        parser = RPN_Parser(rpn,self.general_pset)
        parser.get_tree_structure()
        parser.parse_tree()

        num_record = 0
        data_record = []
        subtree = parser.tree2str(parser.subtree['tree_mode'])

        for root in parser.root['tree_mode']:
            root_expr = parser.tree2str(root)
            if (root_expr.startswith('D')) and (root_expr in subtree):
                subtree = subtree.replace(root_expr, f'DARG_{num_record}')
                data_record.append(root_expr)
                num_record += 1

        for seed in parser.seed['tree_mode']:
            seed_expr = parser.tree2str(seed)
            if seed_expr.startswith('D') and (seed_expr in subtree):
                subtree = subtree.replace(seed_expr, f'DARG_{num_record}')
                data_record.append(seed_expr)
                num_record += 1

        segment1 = data_record
        segment2 = subtree
        segment3 = parser.tree['abbreviation']
        return [segment1, segment2, segment3]

    def toolbox4segment1(self):
        self.pset1 = gp.PrimitiveSetTyped("MAIN", [TypeA] * 5, TypeA)
        name = self.general_toolbox.input2
        for i in range(len(name)):
            self.pset1.renameArguments(**{f'ARG{i}': name[i]})

        creator.create('segment1_Compiler', gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset1)

        self.toolbox1 = base.Toolbox()
        self.toolbox1.register("expr", gp.genHalfAndHalf, pset=self.pset1, min_=1, max_=10)
        self.toolbox1.register('segment1_Compiler', tools.initIterate, getattr(creator, 'segment1_Compiler'),
                               self.toolbox1.expr)
        self.toolbox1.register("population", tools.initRepeat, list, getattr(self.toolbox1, 'segment1_Compiler'))
        self.toolbox1.register("compile", gp.compile, pset=self.pset1)

        for class_name in ['OP_A2A', 'OP_AA2A', 'OP_AG2A', 'OP_AAF2A', 'OP_AF2A']:
            self.add_primitive_byclass(class_name, 'pset1')

        self.pset1.addPrimitive(OP_Closure.id_int, [TypeF], TypeF, name='id_int')
        self.pset1.addPrimitive(OP_Closure.id_float, [TypeG], TypeG, name='id_float')

        for constant_value in [int(i) for i in [1, 2, 3, 5, 10, 20, 30, 60]]:
            self.pset1.addTerminal(constant_value, TypeF)

        for constant_value in [0.05, 0.1]:
            self.pset1.addTerminal(constant_value, TypeG)

    def toolbox4segment2(self, ARG_name):
        self.pset2 = gp.PrimitiveSetTyped("MAIN", [TypeB] * 5 + [TypeD] + [TypeA] * len(ARG_name), TypeA)
        name = self.general_toolbox.input1 + self.general_toolbox.input4 + [f'DARG_{i}' for i in range(len(ARG_name))]
        for i in range(len(name)):
            self.pset2.renameArguments(**{f'ARG{i}': name[i]})

        creator.create('segment2_Compiler', gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset2)

        self.toolbox2 = base.Toolbox()
        self.toolbox2.register("expr", gp.genHalfAndHalf, pset=self.pset2, min_=1, max_=10)
        self.toolbox2.register('segment2_Compiler', tools.initIterate, getattr(creator, 'segment2_Compiler'),
                               self.toolbox2.expr)
        self.toolbox2.register("population", tools.initRepeat, list, getattr(self.toolbox2, 'segment2_Compiler'))
        self.toolbox2.register("compile", gp.compile, pset=self.pset2)

        for class_name in ['OP_B2B', 'OP_BB2B', 'OP_BA2B', 'OP_BG2B', 'OP_BF2B', 'OP_B2D', 'OP_BF2D', 'OP_BA2D',
                           'OP_DD2D', 'OP_B2A', 'OP_BB2A', 'OP_BD2A', 'OP_BBD2A']:
            self.add_primitive_byclass(class_name, 'pset2')

        self.pset2.addPrimitive(OP_Closure.id_int, [TypeF], TypeF, name='id_int')
        self.pset2.addPrimitive(OP_Closure.id_float, [TypeG], TypeG, name='id_float')
        self.pset2.addPrimitive(OP_Closure.id_tensor, [TypeA], TypeA, name='id_tensor')

        for constant_value in [int(i) for i in [1, 2, 3, 5, 10, 20, 30, 60]]:
            self.pset2.addTerminal(constant_value, TypeF)

        for constant_value in [0.05, 0.1]:
            self.pset2.addTerminal(constant_value, TypeG)

    def toolbox4segment3(self):
        self.pset3 = gp.PrimitiveSetTyped("MAIN", [TypeE] + [TypeC] + [TypeA], TypeA)
        self.pset3.renameArguments(**{f'ARG{0}': 'industry'})
        self.pset3.renameArguments(**{f'ARG{1}': 'TypeC_data'})
        self.pset3.renameArguments(**{f'ARG{2}': 'subtree_ARG0'})
        creator.create('segment3_Compiler', gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset3)

        self.toolbox3 = base.Toolbox()
        self.toolbox3.register("expr", gp.genHalfAndHalf, pset=self.pset3, min_=1, max_=10)
        self.toolbox3.register('segment3_Compiler', tools.initIterate, getattr(creator, 'segment3_Compiler'),
                               self.toolbox3.expr)
        self.toolbox3.register("population", tools.initRepeat, list, getattr(self.toolbox3, 'segment3_Compiler'))
        self.toolbox3.register("compile", gp.compile, pset=self.pset3)

        for type in ['A', 'C']:
            class_list = globals()[f"OPclass_name_2{type}"]
            for class_name in class_list:
                self.add_primitive_byclass(class_name, 'pset3')

        self.pset3.addPrimitive(OP_Closure.id_int, [TypeF], TypeF, name='id_int')
        self.pset3.addPrimitive(OP_Closure.id_industry, [TypeE], TypeE, name='id_industry')
        self.pset3.addPrimitive(OP_Closure.id_float, [TypeG], TypeG, name='id_float')

        for constant_value in [int(i) for i in [1, 2, 3, 5, 10, 20, 30, 60]]:
            self.pset3.addTerminal(constant_value, TypeF)

        for constant_value in [0.05, 0.1]:
            self.pset3.addTerminal(constant_value, TypeG)

    def compile_segment1(self, individual):
        compiled_func = self.toolbox1.compile(expr=individual)
        result = compiled_func(*self.Day_data[:])
        return result

    def compile_segment2(self, individual, segment1_data):
        TypeD_data = OP_B2D.Mmask_1h_after_open(self.data_reader.get_Minute_data_daily(self.day_list[0])[0])

        template = torch.full_like(self.Day_data[0], float('nan'))
        for i, day in tqdm(enumerate(self.day_list)):
            curr_Min_data = self.data_reader.get_Minute_data_daily(day)
            curr_Min_data=[i.to(self.device) for i in curr_Min_data]
            curr_Day_data = [data[i].unsqueeze(0) for data in segment1_data]
            curr_data = curr_Min_data + [TypeD_data] + curr_Day_data
            compiled_func = self.toolbox2.compile(expr=individual)
            result = compiled_func(*curr_data[:])
            template[i] = result
        return template

    def compile_segment3(self, individual, segment2_data):
        compiled_func = self.toolbox3.compile(expr=individual)
        data = self.industry_data + self.TypeC_data + [segment2_data]
        result = compiled_func(*data[:])
        return result

    def compile_tree(self, rpn):
        segment1, segment2, segment3 = self.split_segment(rpn)

        self.toolbox4segment1()
        self.toolbox4segment2(segment1)
        self.toolbox4segment3()

        day_data = []
        for segment1 in segment1:
            day_data.append(self.compile_segment1(segment1))

        subtree = self.compile_segment2(segment2, day_data)
        tree = self.compile_segment3(segment3, subtree)
        return tree


if __name__ == "__main__":
    # 测试RPN_Producer类
    producer = RPN_Producer()
    producer.run()
    # print("生成的树：", producer.tree[0])
    # print(producer.subtree[0])

    # # 测试RPN_Compiler类
    # parser = RPN_Parser(producer.tree[0])


    # # compiler.prepare_data([2020])  # 假设准备2020年的数据
    # # compiled_func = compiler.compile(producer.tree[0])  # 编译生成的树中的第一个RPN表达式
    # # print("编译结果：", compiled_func)
    # # print(compiler.rpn)
    # # print(compiler.deap_code)
    # # isinstance('x',gp.Primitive)
    # def print_primitives(pset):
    #     for output_type, primitives in pset.primitives.items():
    #         print(f"Output Type: {output_type.__name__}")
    #         for primitive in primitives:
    #             print(f"  Name: {primitive.name}, Input Types: {[t.__name__ for t in primitive.args]}")
    #             print('#####')
    #             print(primitive.args)


    # def print_terminals(pset):
    #     for output_type, terminals in pset.terminals.items():
    #         print(f"Output Type: {output_type.__name__}")
    #         for terminal in terminals:
    #             print(f"  Terminal: {terminal}")


    # # print_primitives(parser.pset)
    # # print_terminals(parser.pset)
    # # print('####',type(parser.deap_code))
    # tree_structure = parser.get_tree_structure()

    # # 输出树结构
    # # print("Root Node:", tree_structure.root_node.name)
    # # print("Nodes:", [node if isinstance(node, Acyclic_Tree) else node for node in tree_structure.nodes])
    # parser.plot_tree()
    # parser.parse_tree()
    # print(parser.tree)
    # print(parser.subtree)
    # print(parser.branch)
    # print(parser.trunk)
    # print(parser.root)
    # print(parser.seed)
    # print(parser.tree2str(parser.tree['tree_mode']))
