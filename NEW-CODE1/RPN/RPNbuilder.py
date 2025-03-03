import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import re
from RPN.Seed import *
from RPN.Root import *
from RPN.Branch import *
from RPN.Trunk import *
from RPN.Subtree import *
from RPN.Tree import *
from RPN.OrganAbstractClass import *
from ToolsGA.DataReader import *
from tqdm import tqdm
import torch
import networkx as nx
from OP.Info import op_info
import matplotlib.pyplot as plt


class RPN_Producer:
    def __init__(self):
        self.D_OHLC = ['D_O', 'D_H', 'D_L', 'D_C']
        self.D_V = ['D_V']
        self.M_OHLC = ['M_O', 'M_H', 'M_L', 'M_C']
        self.M_V = ['M_V']

    def generate_seed(self):
        DP_Seed_generator = DP_Seed(self.D_OHLC)
        self.dp_seed = DP_Seed_generator.generator(DP_Seed_generator)

        DV_Seed_generator = DV_Seed(self.D_V)
        self.dv_seed = DV_Seed_generator.generator(DV_Seed_generator)

        MP_Seed_generator = MP_Seed(self.M_OHLC)
        self.mp_seed = MP_Seed_generator.generator(MP_Seed_generator)

        MV_Seed_generator = MV_Seed(self.M_V)
        self.mv_seed = MV_Seed_generator.generator(MV_Seed_generator)

    def generate_root(self):
        MP_Root_generator = MP_Root(self.mp_seed)
        self.mp_root = MP_Root_generator.generator(MP_Root_generator)

        DP_Root_generator = DP_Root(self.dp_seed)
        self.dp_root = DP_Root_generator.generator(DP_Root_generator)

        DV_Root_generator = DV_Root(self.dv_seed)
        self.dv_root = DV_Root_generator.generator(DV_Root_generator)

        MV_Root_generator = MV_Root(self.mv_seed)
        self.mv_root = MV_Root_generator.generator(MV_Root_generator)

        self.m_root = self.mv_root + self.mp_root
        self.d_root = self.dv_root + self.dp_root

    def generate_branch(self):
        self.m_branch = []

        M_Branch_MP2D_generator = M_Branch_MP2D(self.mp_seed)
        self.m_branch.extend(M_Branch_MP2D_generator.generator(M_Branch_MP2D_generator))

        M_Branch_MPDP2D_generator = M_Branch_MPDP2D(self.mp_seed, self.dp_seed)
        self.m_branch.extend(M_Branch_MPDP2D_generator.generator(M_Branch_MPDP2D_generator))

        M_Branch_MV2D_generator = M_Branch_MV2D(self.mv_seed)
        self.m_branch.extend(M_Branch_MV2D_generator.generator(M_Branch_MV2D_generator))

        M_Branch_MVDV2D_generator = M_Branch_MVDV2D(self.mv_seed, self.dv_seed)
        self.m_branch.extend(M_Branch_MVDV2D_generator.generator(M_Branch_MVDV2D_generator))

        self.d_branch = []

        D_Branch_DP2C_generator = D_Branch_DP2C(self.dp_root)
        self.d_branch.extend(D_Branch_DP2C_generator.generator((D_Branch_DP2C_generator)))

        D_Branch_DV2C_generator = D_Branch_DV2C(self.dv_root)
        self.d_branch.extend(D_Branch_DV2C_generator.generator((D_Branch_DV2C_generator)))

    def generate_trunk(self):
        Trunk_generator = Trunk(self.m_root, self.d_root, self.d_branch, self.m_branch, 'industry')
        self.trunk = Trunk_generator.generator(Trunk_generator)

    def generate_subtree(self):
        self.subtree = []

        Subtree_withMask_generator = SubtreeWithMask(self.trunk, self.m_branch)
        self.subtree.extend(Subtree_withMask_generator.generator(Subtree_withMask_generator))

        Subtree_noMask_generator = SubtreeNoMask(self.trunk)
        self.subtree.extend(Subtree_noMask_generator.generator(Subtree_noMask_generator))

    def generate_tree(self):
        Tree_generator = Tree(self.subtree)
        self.tree = Tree_generator.generator(Tree_generator)

    def reduce_redundancy(self,rpn):
        deap_code = gp.PrimitiveTree.from_string(rpn, general_pset.pset)
        reduced_code=[]
        for code in deap_code:
            if not code.name.startswith('id'):
                reduced_code.append(code)
        return str(gp.PrimitiveTree(reduced_code))

    def run(self):
        self.generate_seed()
        self.generate_root()
        self.generate_branch()
        self.generate_trunk()
        self.generate_subtree()
        self.generate_tree()
        self.tree=[self.reduce_redundancy(tree) for tree in self.tree]


class RPN_Parser:
    def __init__(self, input, pset=general_pset.pset):
        if type(input) == str:
            self.rpn = input
            self.pset = pset
            self.deap_code = gp.PrimitiveTree.from_string(self.rpn, self.pset)
            self.tree_structure = Acyclic_Tree(self.rpn, self.pset)
        if type(input) == Acyclic_Tree:
            self.tree_structure = input
        self.parse_tree()
        self.graph = nx.DiGraph()
        self.node_counter = {}
        self.node_labels = {}
        
    def get_tree_structure(self):
        return Acyclic_Tree(self.rpn, self.pset)
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

    def tree2str(self,tree):
        depth = self.get_tree_depth(tree)
        abbr, _, _ = self.get_abbrnsub(tree, '', depth)
        return abbr
    
    def tree2lst(self,tree):
        if type(tree) == list:
            return [self.tree2str(x) for x in tree]
        else:
            return self.tree2str(tree)
    def get_graph(self, node,parent=None):
        root = node.root_node.name
        root = self.get_unique_id(root)
        self.graph.add_node(root)
        print(root)
        if parent:
            self.graph.add_edge(parent,root)
        for child in node.nodes:
            if isinstance(child, Acyclic_Tree):
                self.get_graph(child,root)
            else:
                node_name = child.value
                node_name = self.get_unique_id(node_name)
                self.graph.add_node(node_name)
                self.graph.add_edge(root,node_name)
    def get_unique_id(self,name):
        self.node_counter[name] = self.node_counter.get(name,0) + 1
        freq = self.node_counter[name]
        unique_id = f'{name}_{freq - 1}'
        self.node_labels[unique_id] = name
        return unique_id
    def plot_tree(self):
        self.get_graph(self.tree_structure)
        plt.figure(figsize=(30, 20))
        pos = nx.nx_agraph.graphviz_layout(self.graph, prog="dot")  # 使用 Graphviz 布局
        nx.draw(self.graph, pos,labels = self.node_labels, with_labels=True, node_color="lightblue", node_size=2000, font_size=12, edge_color="gray")
        plt.show()


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

    def tree2dict_abbr(self):
        self.tree_dict = {
            'tree': [self.tree['abbreviation']],
            'subtree': [self.subtree['abbreviation']],
            'branch': self.branch['abbreviation'],
            'trunk': self.trunk['abbreviation'],
            'root': self.root['abbreviation'],
            'seed': [self.argnorm(seed) for seed in self.seed['abbreviation']]
        }
        return self.tree_dict
    def tree2dict_full(self):
        full_dict = {
            'tree': self.tree2lst(self.tree['tree_mode']),
            'subtree': self.tree2lst(self.subtree['tree_mode']),
            'branch': self.tree2lst(self.branch['tree_mode']),
            'trunk': self.tree2lst(self.trunk['tree_mode']),
            'root': self.tree2lst(self.root['tree_mode']),
            'seed': self.tree2lst(self.seed['tree_mode'])
        }
        return full_dict 

class RPN_Compiler:
    def __init__(self, year_list, device=torch.device("cuda")):

        self.year_list = year_list
        self.general_pset = general_pset.pset
        self.device = device
        self.__init_data(self.year_list)

    def __init_data(self, year_list):
        self.data_reader = MmapReader()
        self.day_list = self.data_reader.get_daylist(year_list)
        self.D_O, self.D_H, self.D_L, self.D_C, self.D_V = list(self.data_reader.get_Day_data(year_list))
        self.industry = [self.data_reader.get_Barra(year_list)[:, :, 10:].to(self.device)]

    def extract_op(self, expression):
        op_list = []
        # 初始化一个列表，用于存储左括号、右括号和逗号的位置
        positions = []
        # 初始化括号计数器
        bracket_count = 0
        # 遍历表达式字符串
        for i, char in enumerate(expression):
            if char == '(':
                # 如果是左括号，计数器加1
                bracket_count += 1
                if bracket_count == 1:
                    # 记录第一个左括号的位置
                    positions.append(i)
            elif char == ')':
                # 如果是右括号，计数器减1
                bracket_count -= 1
                if bracket_count == 0:
                    # 如果括号计数器为0，说明找到了匹配的右括号，记录位置
                    positions.append(i)
            elif char == ',' and bracket_count == 1:
                # 如果是逗号且括号计数器为1，记录逗号位置
                positions.append(i)

        # 如果没有记录任何位置，说明这段表达式中没有算子
        if not positions:
            return op_list

        # 提取第一个左括号之前的所有字符，即算子名称
        op_name = expression[:positions[0]]
        # 将算子名称添加到结果列表中
        op_list.append(op_name)

        # 如果有多个位置记录，说明有嵌套的算子
        if len(positions) > 1:
            # 遍历所有记录的位置，提取子表达式并递归调用
            for start, end in zip(positions[:-1], positions[1:]):
                # 提取子表达式
                sub_expression = expression[start + 1:end]
                # 递归调用函数，传入空列表
                sub_list = self.extract_op(sub_expression)
                # 将递归调用的结果扩展到主列表中
                op_list.extend(sub_list)
        return op_list

    def add_op_class(self, op):
        interface = op_info[op.strip()]['classification']['interface']
        return f"OP_{interface['属'][:-1]}2{interface['目'][:-1]}.{op}"

    def replace_primities(self, rpn):
        used_op = self.extract_op(rpn)
        used_op = [i.strip() for i in used_op]
        used_op = list(dict.fromkeys(used_op))
        for op in used_op:
            rpn = rpn.replace(op, self.add_op_class(op))
        return rpn

    def replace_D_tensor(self, rpn):
        count = 0
        pattern = r"D_tensor"

        def replacer(match):
            nonlocal count  # 使用 nonlocal 关键字访问外部的 count 变量
            current_count = count  # 保存当前计数
            count += 1  # 计数器递增
            return f"D_tensor{current_count}"  # 返回替换后的字符串 D_i

        result = re.sub(pattern, replacer, rpn)
        return result

    def compile_module1(self, rpn, D_tensor: [torch.Tensor]):
        rpn = self.replace_D_tensor(rpn)
        rpn = self.replace_primities(rpn)

        for i in range(len(D_tensor)):
            locals()[f'D_tensor{i}'] = D_tensor[i].to(self.device)

        return eval(rpn)

    def compile_module2(self, rpn, D_tensor: [torch.Tensor]):

        rpn = self.replace_D_tensor(rpn)
        rpn = self.replace_primities(rpn)

        for i in range(len(D_tensor)):
            locals()[f'D_tensor_all{i}'] = D_tensor[i]

        template = torch.full((len(self.day_list), len(self.data_reader.StockCodes)), float('nan'))
        for i, day in tqdm(enumerate(self.day_list)):
            M_O, M_H, M_L, M_C, M_V = self.data_reader.get_Minute_data_daily(day)
            M_O, M_H, M_L, M_C, M_V = [i.to(self.device) for i in [M_O, M_H, M_L, M_C, M_V]]
            for j in range(len(D_tensor)):
                locals()[f'D_tensor{j}'] = locals()[f'D_tensor_all{j}'][i].to(self.device)
            template[i] = eval(rpn)

        return template

    def adjust_memorizer(self, deap_primitive, string_memorizer):
        expr = f"{deap_primitive.name}({', '.join(string_memorizer[:deap_primitive.arity])})"
        string_memorizer = string_memorizer[deap_primitive.arity:]
        string_memorizer.insert(0, expr)
        return string_memorizer

    def compile(self, rpn):
        name = general_pset.input1 + general_pset.input2 + general_pset.input3 + general_pset.input4 + general_pset.input5
        deap_code = gp.PrimitiveTree.from_string(rpn, self.general_pset)
        deap_code.reverse()
        D_tensor_memorizer = []
        string_memorizer = []
        for code in deap_code:
            if isinstance(code, gp.Terminal):

                if code.name.startswith('ARG') and int(code.name[3:]) >= 5:
                    D_tensor_memorizer.insert(0, getattr(self, name[int(code.name[3:])]))
                    string_memorizer.insert(0, 'D_tensor')

                elif code.name.startswith('ARG') and int(code.name[3:]) < 5:
                    string_memorizer.insert(0, name[int(code.name[3:])])

                else:
                    string_memorizer.insert(0, code.name)

            if isinstance(code, gp.Primitive):
                if code.name.startswith('D'):
                    string_memorizer = self.adjust_memorizer(code, string_memorizer)
                    flag = any(item in string_memorizer[0] for item in name[:5])

                    if flag == 0:
                        count = string_memorizer[0].count("D_tensor")
                        result = self.compile_module1(string_memorizer[0], D_tensor_memorizer[:count])
                        D_tensor_memorizer = D_tensor_memorizer[count:]
                        D_tensor_memorizer.insert(0, result)
                        string_memorizer[0] = 'D_tensor'

                    elif flag == 1:
                        count = string_memorizer[0].count("D_tensor")
                        result = self.compile_module2(string_memorizer[0], D_tensor_memorizer[:count])
                        D_tensor_memorizer = D_tensor_memorizer[count:]
                        D_tensor_memorizer.insert(0, result)
                        string_memorizer[0] = 'D_tensor'

                elif code.name.startswith('M'):
                    string_memorizer = self.adjust_memorizer(code, string_memorizer)

        return D_tensor_memorizer[0]


if __name__ == "__main__":
    # 测试RPN_Producer类
    producer = RPN_Producer()
    producer.run()
    print("生成的树：", producer.tree[0])

    

    # parser = RPN_Parser(producer.tree[0])
    atree = Acyclic_Tree(producer.tree[0],general_pset.pset)
    parser = RPN_Parser(atree)

    # tree_structure = parser.tree_structure

    # print("Root Node:")
    # print(tree_structure.root_node.name)

    # print("Nodes (Each node is either an Acyclic_Tree instance or a value):")
    # print([node if isinstance(node, Acyclic_Tree) else node for node in tree_structure.nodes])

    # print("Tree Structure:")
    # print(parser.tree)

    # print("Subtree:")
    # print(parser.subtree)

    # print("Branch:")
    # print(parser.branch)

    # print("Trunk:")
    # print(parser.trunk)

    # print("Root:")
    # print(parser.root)

    # print("Seed:")
    # print(parser.seed)

    # print("Tree to Abbreviation Dict:")
    # print(parser.tree2dict_abbr())

    # print("Tree to Full Dict:")
    # print(parser.tree2dict_full())

    print("Plotting the Tree:")
    parser.plot_tree()

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


    # print_primitives(parser.pset)
    # print_terminals(parser.pset)
    # print('####',type(parser.deap_code))

    # 输出树结构
    # print("Root Node:", tree_structure.root_node.name)
    # print("Nodes:", [node if isinstance(node, Acyclic_Tree) else node for node in tree_structure.nodes])
