import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

from ToolsGA import *
from OP import *
from RPN import *
import random
from collections import defaultdict
from deap import gp
import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz

class config:
    seed_size=10
    root_size=10
    branch_size=10
    trunk_size=10
    subtree_size=10
    tree_size=10
    default_year = [2016]

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
        self.tree_structure = None
        self.tree = None

    def get_abbrnsub(self, ac_tree, substr, flag=0, count=0):
        flag = max(flag-1,0)
        tree_depth = self.get_tree_depth(ac_tree)
        abbr = ac_tree.root_node.name + '('
        sub = []
        for i,node in enumerate(ac_tree.nodes):
            if i > 0:
                abbr += ', '
            if isinstance(node, Acyclic_Tree):
                if flag+1+self.get_tree_depth(node)-tree_depth <= 0:
                    abbr += f'{substr}_ARG{count}'
                    count += 1
                    sub.append(node)
                else:
                    sub_abbr,sub_sub,count = self.get_abbrnsub(node,substr,flag+1+self.get_tree_depth(node)-tree_depth,count)
                    abbr += sub_abbr
                    sub.extend(sub_sub)

            elif isinstance(node, gp.Terminal):
                abbr += str(node.value)
            else:
                raise ValueError('instance error')
        return abbr+')',sub,count
        
    def get_tree_depth(self,tree):
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

    def print_tree(self, node=None, level=0):
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
                self.print_tree(child, level + 1)
            else:
                print(f"    " * (level + 1)+ str(child.value))


    def add_nodes_edges(self, tree, parent=None,level=0,count=0):
        if isinstance(tree, Acyclic_Tree):
            for i, child in enumerate(tree.nodes):
                count+=1
                if isinstance(child, Acyclic_Tree):
                    child_label = f"{child.root_node.name}"#+str(level)+str(i)
                    for _ in range(count):
                        child_label += '\u200b'
                    self.G.add_node(child_label)
                    if parent:
                        self.G.add_edge(parent, child_label)
                    count = self.add_nodes_edges(child, child_label,level+1,count)
                else:
                    child_label = str(child.value)
                    for _ in range(count):
                        child_label += '\u200b'
                    self.G.add_node(child_label)
                    if parent:
                        self.G.add_edge(parent, child_label)
        return count

    def plot_tree(self, node=None, level=0):
        if self.tree_structure is None:
            self.get_tree_structure()
    
        if node is None:
            node = self.tree_structure
    
        # 创建一个空的有向图
        self.G = nx.DiGraph()
    
        # 添加根节点
        root_label = f"{node.root_node.name}"
        self.G.add_node(root_label)
        self.add_nodes_edges(node, root_label)
    
        # 使用matplotlib绘制图形
        pos = nx.drawing.nx_agraph.graphviz_layout(self.G, prog='dot')
        plt.figure(figsize=(12, 8))
        nx.draw(self.G, pos, with_labels=True, arrows=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold')
        plt.show()


    def parse_from_the_outside_in(self):
        pass

    def parse_from_the_inside_out(self):
        pass

    def parse_tree(self):
        abbr_tree,sub_tree,_ = self.get_abbrnsub(self.tree_structure,'subtree')
        self.tree={
            'abbreviation': abbr_tree,
            'tree_mode':self.tree_structure,
        }

        abbr_subtree = ''
        branch_lst = []
        trunk_lst = []
        for tree in sub_tree:
            count_branch = 0
            count_trunk = 0
            abbr_subtree += tree.root_node.name + '('
            for i,node in enumerate(tree.nodes):
                if i > 0:
                    abbr_subtree += ', '
                if isinstance(node, Acyclic_Tree):
                    if i < len(tree.nodes)-1 or 'TypeD' not in [t.__name__ for t in tree.root_node.args]:
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
            
        self.subtree={
            'abbreviation': abbr_subtree,
            'tree_mode':sub_tree[0],
        }


        abbr_trunk = []
        sub_trunk = []
        count = 0
        for trunk in trunk_lst:
            trunk_depth = self.get_tree_depth(trunk)-2
            abbr,sub,count = self.get_abbrnsub(trunk,'root',trunk_depth,count)
            abbr_trunk.append(abbr)
            sub_trunk.extend(sub)
        
        self.trunk = {
            'abbreviation':abbr_trunk,
            'tree_mode':trunk_lst,
        }
        count = 0
        abbr_root = []
        sub_root = []
        for root in sub_trunk:
            abbr,sub,count = self.get_abbrnsub(root,'seed',0,count)
            abbr_root.append(abbr)
            sub_root.extend(sub)

        abbr_branch = []
        for branch in branch_lst:
            branch_depth = self.get_tree_depth(branch)-1
            abbr,sub,count = self.get_abbrnsub(branch,'seed',branch_depth,count)
            abbr_branch.append(abbr)
            sub_root.extend(sub)

        self.branch={
            'abbreviation': abbr_branch,
            'tree_mode': branch_lst,
        }

        self.root = {
            'abbreviation':abbr_root,
            'tree_mode':sub_trunk,
        }

        self.seed = {
            'abbreviation':[seed.abbreviation for seed in sub_root],
            'tree_mode':sub_root,
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
            'tree':[self.tree['abbreviation']],
            'subtree':[self.subtree['abbreviation']],
            'branch':self.branch['abbreviation'],
            'trunk':self.trunk['abbreviation'],
            'root':self.root['abbreviation'],
            'seed':[self.argnorm(seed) for seed in self.seed['abbreviation']]
        }
        return self.tree_dict



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

class GeneticAlgorithm:
    def __init__(self, ops_lists, population_size=10, crossover_prob=0.5, mutation_prob=0.2, generations=1):
        self.ops_lists = ops_lists
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.generations = generations

        # 初始化DEAP工具箱
        self.toolbox = base.Toolbox()

        # 创建适应度类和个体类
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # 注册个体生成函数
        self.toolbox.register("individual", self.init_individual, creator.Individual, self.ops_lists)

        # 注册种群生成函数
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # 注册适应度评估函数
        self.toolbox.register("evaluate", self.eval_fitness)

        # 注册自定义变异函数
        self.toolbox.register("mutate", self.custom_mutate)

        # 注册自定义交叉函数
        self.toolbox.register("mate1", self.custom_crossover1)
        self.toolbox.register("mate2", self.custom_crossover2)

        # 注册选择操作
        self.toolbox.register("select", tools.selBest)

    def init_individual(self, icls, content):
        # 从每个子列表中随机选择一个索引
        return icls(random.randint(0, len(lst) - 1) for lst in content)

    def eval_fitness(self, individual):
        # 所有个体的适应度值都为1
        return (1.0,)

    def custom_mutate(self, individual, min_nodes=0, max_nodes=2):
        # 从指定范围(min_nodes, max_nodes)中随机选择变异节点的个数
        num_nodes_to_mutate = random.randint(min_nodes, max_nodes)
        # 从所有节点中随机选择相应个数的不同节点
        nodes_to_mutate = random.sample(range(len(individual)), num_nodes_to_mutate)
        # 对选中的节点实施变异
        for node in nodes_to_mutate:
            individual[node] = random.randint(0, len(self.ops_lists[node]) - 1)
        return individual,

    def custom_crossover1(self, ind1, ind2, min_nodes=1, max_nodes=2):
        # 从指定范围(min_nodes, max_nodes)中随机选择交叉节点的个数
        num_nodes_to_crossover = random.randint(min_nodes, max_nodes)
        # 从所有节点中随机选择相应个数的不同节点
        nodes_to_crossover = random.sample(range(len(ind1)), num_nodes_to_crossover)
        # 对选中的节点进行交叉
        for node in nodes_to_crossover:
            ind1[node], ind2[node] = ind2[node], ind1[node]
        return ind1, ind2

    def custom_crossover2(self, ind1, ind2):
        # 随机选择一个节点
        crossover_point = random.randint(0, len(ind1) - 1)
        # 从该节点开始，交换两个个体的后续部分
        ind1[crossover_point:], ind2[crossover_point:] = ind2[crossover_point:], ind1[crossover_point:]
        return ind1, ind2

    def run(self):
        # 初始化种群
        pop = self.toolbox.population(n=self.population_size)
        print("Initial population:", pop)#仅供演示

        # 评估初始种群
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for gen in range(self.generations):
            # 选择下一代个体
            offspring = self.toolbox.select(pop, len(pop))
            # 克隆选中的个体
            offspring = list(map(self.toolbox.clone, offspring))

            # 应用交叉操作
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    # 随机选择一种交叉方式
                    if random.random() < 0.5:
                        self.toolbox.mate1(child1, child2, min_nodes=1, max_nodes=2)
                    else:
                        self.toolbox.mate2(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # 应用自定义变异操作
            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant, min_nodes=0, max_nodes=2)
                    del mutant.fitness.values

            # 评估无效适应度的个体
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # 替换种群
            pop[:] = offspring

        return pop

class Acyclic_Tree:
    def __init__(self, deap_str, pset):
        self.tree = deap_str
        self.root_node = None  # gp.Primitive or gp.Terminal
        self.nodes = []  # List[Union[Leaf, Acyclic_Tree]]
        self.abbreviation = ''
        self.pset = pset

        # 解析DEAP字符串并构建树结构
        self.parse_deap_str(deap_str, pset)

    def parse_deap_str(self, deap_str, pset):
        # 将DEAP字符串转换为PrimitiveTree
        primitive_tree = gp.PrimitiveTree.from_string(deap_str, pset)
        # 递归构建树结构
        self.root_node, self.nodes = self.build_tree(primitive_tree)
        self.abbreviation += self.root_node.name if isinstance(self.root_node, gp.Primitive) else str(self.root_node.value)
        self.abbreviation += '('
        counter = 0
        for i,node in enumerate(self.nodes):
            if i > 0:
                self.abbreviation += ', '
            if isinstance(node, Acyclic_Tree):
                self.abbreviation += f"ARG{counter}" 
                counter += 1
            else:
                self.abbreviation += str(node.value)   
        self.abbreviation += ')'         

    def build_tree(self, primitive_tree):
        root = primitive_tree[0]  # 根节点
        nodes = []
        deap_code = self.extract_string(self.tree)
        if isinstance(root, gp.Primitive):
            # 如果根节点是操作（primitive），递归构建子树
            for i in range(root.arity):
                if '(' in deap_code[i]:
                    child_tree = Acyclic_Tree(deap_code[i], self.pset)
                    nodes.append(child_tree)
                else:
                    nodes.append(gp.PrimitiveTree.from_string(deap_code[i], self.pset)[0])
        else:
            # 如果根节点是终端（terminal），直接返回
            return root, []

        return root, nodes
    

    def extract_string(self,s):
    # 检查字符串是否包含括号
        if '(' not in s:
            return None

        # 初始化计数器和临时字符串
        count = 0
        temp_str = ''
        result_list = []
        
        # 遍历字符串中的每个字符
        for char in s:
            if char == '(':
                count += 1
                # 如果是第一个左括号，跳过它
                if count == 1:
                    continue
            elif char == ')':
                count -= 1
                # 如果是与第一个左括号匹配的右括号，处理临时字符串并停止遍历
                if count == 0:
                    # 分割临时字符串并添加到结果列表
                    # 只有在所有左括号都匹配的情况下才进行分割
                    if temp_str:
                        result_list.append(temp_str.strip())
                    break
            # 如果在最外层括号内，处理字符
            if count > 0:
                if char == ',' and count == 1:
                    # 如果遇到逗号且所有左括号都已匹配，添加之前的临时字符串到结果列表
                    if temp_str:
                        result_list.append(temp_str.strip())
                        temp_str = ''
                elif char != ' ' or temp_str:
                    temp_str += char
        
        
        return result_list

    
    

if __name__ == "__main__":
    # 测试RPN_Producer类
    producer = RPN_Producer()
    producer.run()
    print("生成的树：", producer.tree[0])
    parser = RPN_Parser(producer.tree[0])
    tree_structure = parser.get_tree_structure()
    parser.print_tree()
    parser.plot_tree()
    print(parser.tree2dict())
    # 编译
    compiler = RPN_Compiler()
    compiler.prepare_data(config.default_year)
    print(compiler.compile('D_ts_mean(D_O, 5)'))
    # 输入的操作列表
    ops_lists = [
        ["op1", "op2", "op3", '', ''],
        ["op4", "op5", "op6", '', ''],
        ["op7", "op8", "op9", "op10"],
        ["op7", "op8", "op9", "op10"],
        ["op7", "op8", "op9", "op10", 'a', '']
    ]

    # 初始化遗传算法类
    ga = GeneticAlgorithm(ops_lists, population_size=10, crossover_prob=0.5, mutation_prob=0.2, generations=1)

    # 运行遗传算法
    pop = ga.run()

    # 输出最终种群
    print("Final population:")
    for ind in pop:
        print(ind, ind.fitness.values)
