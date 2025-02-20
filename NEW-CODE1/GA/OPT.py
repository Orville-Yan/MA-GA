from RPN import *
from ToolsGA import *
import random

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



class GA_optimizer:
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