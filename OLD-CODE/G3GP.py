import 日频数据的导入和预处理工具 as datatools
import numpy as np
from deap import base
from deap import creator
from deap import tools
from functools import partial

from GP_tools import *
from 回测框架 import stratified, indicators
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda")

class config:
    tree_size = 10000
    subtree_size = 1000
    mutation_probability = 0.2
    crossover_probability = 0.8
    num_generations = 3
    min_tree_depth = 1
    max_tree_depth = 1
    predict_period = 5
    root_size = 3000
    standard=0.06
    train_start_time, train_end_time = 2016, 2018
    predict_start_time, predict_end_time = 2019, 2021
    fitness_func1 = back_test
    formula_tree_path = "C:\\Users\\74989\\Desktop\\factor_formula_tree.xlsx"
    root_operators = [ 'ts_mean', 'ts_middle_mean', 'ts_harmonic_mean', 'ts_weight_mean', 'ts_mask']

class data_process:
    def __init__(self):
        self.open, self.close, self.high, self.low = [torch.tensor(datatools.open, dtype=torch.float32, device=device),
                                                      torch.tensor(datatools.close, dtype=torch.float32, device=device),
                                                      torch.tensor(datatools.high, dtype=torch.float32, device=device),
                                                      torch.tensor(datatools.low, dtype=torch.float32, device=device)]
        self.open, self.close, self.high, self.low = [torch.where(i < 1e-5, float('nan'), i) for i in
                                                      [self.open, self.close, self.high, self.low]]
        self.volume = torch.tensor(datatools.get_volume().values, dtype=torch.float32, device=device)
        self.volume = torch.where(self.volume < 1e-5, float('nan'), self.volume)
        self.train_start_time = config.train_start_time
        self.train_end_time = config.train_end_time
        self.predict_start_time = config.predict_start_time
        self.predict_end_time = config.predict_end_time

    def get_index(self):
        date_series = pd.Series(datatools.TradingDate)
        self.train_index = date_series.index[
            (date_series.dt.year >= self.train_start_time) & (date_series.dt.year <= self.train_end_time)]
        self.predict_index = date_series.index[
            (date_series.dt.year >= self.predict_start_time) & (date_series.dt.year <= self.predict_end_time)]

    def get_compile_list(self):
        s = list(self.train_index) + list(self.predict_index)
        self.bt_list = [self.open[s], self.close[s], self.high[s], self.low[s], self.volume[s]]
        self.train_list = [self.open[self.train_index], self.close[self.train_index], self.high[self.train_index],
                           self.low[self.train_index], self.volume[self.train_index]]
        self.predict_list = [self.open[self.predict_index], self.close[self.predict_index],
                             self.high[self.predict_index], self.low[self.predict_index],
                             self.volume[self.predict_index]]

    def get_labels(self):
        pct_change = op.ts_delay(self.close, -config.predict_period) / op.ts_delay(self.open, -1)
        pct_change = torch.where((pct_change == torch.inf) | (pct_change == -torch.inf), float('nan'), pct_change)
        jump_open = (op.ts_delay(self.open, -1) / self.close) >= 1.095
        pct_change = torch.where(jump_open, float('nan'), pct_change)
        self.train_target = pct_change[self.train_index].to(device)
        self.predict_target = pct_change[self.predict_index].to(device)

        status20 = op.ts_mean(torch.from_numpy((1 - datatools.ST) * (datatools.status)), 20) > 0.5
        listed = torch.from_numpy(datatools.ListedDate >= 60)
        self.clean = ~(listed.to(device) * status20.to(device))

    def get_stocks_count(self):
        df = datatools.get_pct_change()
        stocks_count = df.apply(lambda x: len(x.dropna()), axis=1).values
        self.train_stocks_count = torch.from_numpy(stocks_count[self.train_index]).to(device)

    def get_barra(self):
        self.barra_train = get_barra(range(config.train_start_time, config.train_end_time + 1)).to(device)
        self.barra_predict = get_barra(range(config.predict_start_time, config.predict_end_time + 1)).to(device)
        self.dict = torch.load('dict.pt')['name']

    def get_basic_data(self):
        self.get_index()
        self.get_stocks_count()
        self.get_compile_list()
        self.get_labels()
        self.get_barra()

class Generate_Tree(data_process):
    def __init__(self):
        super().__init__()
        self.root_size = config.root_size
        self.subtree_size = config.subtree_size

        self.no_fitness=0
        self.gene={}
        self.error=[]

        self.tree_size = config.tree_size
        self.mutation_probability = config.mutation_probability
        self.crossover_probability = config.crossover_probability
        self.num_generations = config.num_generations
        self.max_tree_depth = config.max_tree_depth
        self.min_tree_depth = config.min_tree_depth
        self.formula_tree_path = config.formula_tree_path

    def fitness(self, individual):
        if str(individual) in self.gene.keys():
            return self.gene[str(individual)]

        else:
            self.no_fitness+=1
            try:
                compiled_func = self.toolbox4.compile_train(expr=individual)
                result = compiled_func(*self.train_list[:])
                num_count = torch.sum(~torch.isnan(result), dim=-1) / self.train_stocks_count
                num_count = torch.where(num_count == 0, float('nan'), num_count)

                if nanmean(num_count) < 0.8:
                    return (0,)
                else:
                    cum_return, interval_num = config.fitness_func1(result, self.train_target, self.clean[self.train_index])
                    fitness_1 = cum_return[0] ** (50 / interval_num) - 1
                    fitness_2 = cum_return[-1] ** (50 / interval_num) - 1
                    fitness_mean = (torch.mean(cum_return) ** (50 / interval_num) - 1).item()
                    s = abs(max([fitness_1.item() - fitness_mean, fitness_2.item() - fitness_mean]))
                    if np.isnan(s) | (interval_num == 0):
                        return (0,)
                    else:
                        return (s,)

            except:
                self.error.append(individual)
                return (0,)

    def generate_toolbox1(self):
        self.pset1 = gp.PrimitiveSetTyped("MAIN", [torch.Tensor] * len(self.train_list), torch.Tensor)

        name = ['open', 'close', 'high', 'low', 'volume']
        for i in range(len(self.train_list)):
            self.pset1.renameArguments(**{f'ARG{i}': name[i]})

        operator = op()

        for func_name in config.root_operators[:-2]:
            func = getattr(operator, func_name)
            self.pset1.addPrimitive(func, [torch.Tensor, int], torch.Tensor, name=func_name)

        for part in [0, 1, 2, 3]:
            for method in ['mean', 'weight_mean']:
                func = partial(getattr(operator, 'ts_mask'), part=part, method=method)
                self.pset1.addPrimitive(func, [torch.Tensor, torch.Tensor, int], torch.Tensor,
                                        name='ts_mask' + str(part) + '_' + method)

        int_values = [int(i) for i in [1, 2, 3, 5, 10, 20, 60]]
        for constant_value in int_values:
            self.pset1.addTerminal(constant_value, int)

        constant_function = operator.create_constant_function(2)
        self.pset1.addPrimitive(constant_function, [torch.Tensor], int, name='two')
        ts_weight_mean_func = getattr(operator, 'ts_weight_mean')
        self.pset1.addPrimitive(ts_weight_mean_func, [torch.Tensor, torch.Tensor, int], torch.Tensor,
                                name='ts_weight_mean')

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Root", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset1)

        self.toolbox1 = base.Toolbox()
        self.toolbox1.register("expr", gp.genHalfAndHalf, pset=self.pset1, min_=1, max_=1)
        self.toolbox1.register("Root", tools.initIterate, creator.Root, self.toolbox1.expr)
        self.toolbox1.register("population", tools.initRepeat, list, self.toolbox1.Root)
        self.toolbox1.register("compile", gp.compile, pset=self.pset1)

    def generate_toolbox2(self):
        self.pset2 = gp.PrimitiveSetTyped("MAIN", [torch.Tensor] * len(self.root_code), torch.Tensor)
        operator = op()
        standardize_tool = standardize()
        for func_name in standardize_tool.op_name[:4]:
            func = getattr(standardize_tool, func_name)
            self.pset2.addPrimitive(func, [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor)

        for func_name in standardize_tool.op_name[5:]:
            func = getattr(standardize_tool, func_name)
            self.pset2.addPrimitive(func, [torch.Tensor, int], torch.Tensor)

        int_values = [int(i) for i in [2, 3, 5, 10, 20, 60]]
        for constant_value in int_values: self.pset2.addTerminal(constant_value, int)

        constant_function = operator.create_constant_function(2)
        self.pset2.addPrimitive(constant_function, [torch.Tensor], int, name='two')
        self.pset2.addPrimitive(getattr(standardize_tool, standardize_tool.op_name[4]), [torch.Tensor, torch.Tensor],
                                torch.Tensor, name=standardize_tool.op_name[4])

        creator.create("subtree", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset2)

        self.toolbox2 = base.Toolbox()
        self.toolbox2.register("expr", gp.genHalfAndHalf, pset=self.pset2, min_=1, max_=1)
        self.toolbox2.register("subtree", tools.initIterate, creator.subtree, self.toolbox2.expr)
        self.toolbox2.register("population", tools.initRepeat, list, self.toolbox2.subtree)
        self.toolbox2.register("compile", gp.compile, pset=self.pset2)

    def generate_toolbox3(self):
        self.pset3_train = gp.PrimitiveSetTyped("MAIN", [torch.Tensor] * len(self.subtree_str), torch.Tensor)
        self.pset3_predict = gp.PrimitiveSetTyped("MAIN", [torch.Tensor] * len(self.subtree_str), torch.Tensor)

        operator = op()
        for i, barra_name in enumerate(self.dict[:10]):
            func_train = partial(getattr(operator, 'cs_barra_neut'), barra_factor=self.barra_train[:, :, i])
            self.pset3_train.addPrimitive(func_train, [torch.Tensor], torch.Tensor, name='cs_' + barra_name + '_neut')
            func_predict = partial(getattr(operator, 'cs_barra_neut'), barra_factor=self.barra_predict[:, :, i])
            self.pset3_predict.addPrimitive(func_predict, [torch.Tensor], torch.Tensor,
                                            name='cs_' + barra_name + '_neut')

        func_train = partial(getattr(operator, 'cs_ind_neut'), ind=self.barra_train[:, :, 10:])
        self.pset3_train.addPrimitive(func_train, [torch.Tensor], torch.Tensor, name='cs_ind_neut')
        func_predict = partial(getattr(operator, 'cs_ind_neut'), ind=self.barra_predict[:, :, 10:])
        self.pset3_predict.addPrimitive(func_predict, [torch.Tensor], torch.Tensor, name='cs_ind_neut')

        for func_name in operator.variable1_func_list:
            func = getattr(operator, func_name)
            self.pset3_train.addPrimitive(func, [torch.Tensor], torch.Tensor)
            self.pset3_predict.addPrimitive(func, [torch.Tensor], torch.Tensor)

        for func_name in operator.variable2_func_list:
            func = getattr(operator, func_name)
            self.pset3_train.addPrimitive(func, [torch.Tensor, torch.Tensor], torch.Tensor)
            self.pset3_predict.addPrimitive(func, [torch.Tensor, torch.Tensor], torch.Tensor)

        for func_name in operator.variable3_func_list:
            func = getattr(operator, func_name)
            self.pset3_train.addPrimitive(func, [torch.Tensor, torch.Tensor, int], torch.Tensor)
            self.pset3_predict.addPrimitive(func, [torch.Tensor, torch.Tensor, int], torch.Tensor)

        for func_name in operator.on_parameter_func_list:
            func = getattr(operator, func_name)
            self.pset3_train.addPrimitive(func, [torch.Tensor, int], torch.Tensor, name=func_name)
            self.pset3_predict.addPrimitive(func, [torch.Tensor, int], torch.Tensor, name=func_name)

        for part in [0, 1, 2, 3]:
            for method in ['mean', 'std', 'weight_mean', 'prod']:
                func = partial(getattr(operator, 'ts_mask'), part=part, method=method)
                self.pset3_train.addPrimitive(func, [torch.Tensor, torch.Tensor, int], torch.Tensor,
                                              name='ts_mask' + str(part) + '_' + method)
                self.pset3_predict.addPrimitive(func, [torch.Tensor, torch.Tensor, int], torch.Tensor,
                                                name='ts_mask' + str(part) + '_' + method)

        int_values = [int(i) for i in [2, 3, 5, 10, 20, 60]]
        for constant_value in int_values:
            self.pset3_train.addTerminal(constant_value, int)
            self.pset3_predict.addTerminal(constant_value, int)

        constant_function = operator.create_constant_function(2)
        self.pset3_train.addPrimitive(constant_function, [torch.Tensor], int, name='two')
        self.pset3_predict.addPrimitive(constant_function, [torch.Tensor], int, name='two')

        creator.create("tree_train", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset3_train)
        creator.create("tree_predict", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset3_predict)
        self.toolbox3 = base.Toolbox()
        self.toolbox3.register("expr_train", gp.genHalfAndHalf, pset=self.pset3_train, min_=self.min_tree_depth,
                               max_=self.max_tree_depth)
        self.toolbox3.register("expr_predict", gp.genHalfAndHalf, pset=self.pset3_predict, min_=self.min_tree_depth,
                               max_=self.max_tree_depth)
        self.toolbox3.register("tree_train", tools.initIterate, creator.tree_train, self.toolbox3.expr_train)
        self.toolbox3.register("tree_predict", tools.initIterate, creator.tree_predict, self.toolbox3.expr_predict)

        self.toolbox3.register("population_train", tools.initRepeat, list, self.toolbox3.tree_train)
        self.toolbox3.register("population_predict", tools.initRepeat, list, self.toolbox3.tree_predict)
        self.toolbox3.register("compile_train", gp.compile, pset=self.pset3_train)
        self.toolbox3.register("compile_predict", gp.compile, pset=self.pset3_predict)

        self.toolbox3.register("evaluate", self.fitness)
        self.toolbox3.register("select1", tools.selTournament, tournsize=2)
        self.toolbox3.register("select2", tools.selRoulette, k=10)
        self.toolbox3.register("mate1", gp.cxOnePoint)

        self.toolbox3.register("mutate1", gp.mutShrink)
        self.toolbox3.register("mutate2", gp.mutUniform, expr=self.toolbox3.expr_train, pset=self.pset3_train)
        self.toolbox3.register("mutate3", gp.mutNodeReplacement, pset=self.pset3_train)

    def generate_toolbox4(self):
        self.pset4_train = gp.PrimitiveSetTyped("MAIN", [torch.Tensor] * len(self.train_list), torch.Tensor)
        self.pset4_predict = gp.PrimitiveSetTyped("MAIN", [torch.Tensor] * len(self.predict_list), torch.Tensor)
        name = ['open', 'close', 'high', 'low', 'volume']
        for i in range(len(self.train_list)):
            self.pset4_train.renameArguments(**{f'ARG{i}': name[i]})
            self.pset4_predict.renameArguments(**{f'ARG{i}': name[i]})
        operator = op()

        for i, barra_name in enumerate(self.dict[:10]):
            func_train = partial(getattr(operator, 'cs_barra_neut'), barra_factor=self.barra_train[:, :, i])
            self.pset4_train.addPrimitive(func_train, [torch.Tensor], torch.Tensor, name='cs_' + barra_name + '_neut')
            func_predict = partial(getattr(operator, 'cs_barra_neut'), barra_factor=self.barra_predict[:, :, i])
            self.pset4_predict.addPrimitive(func_predict, [torch.Tensor], torch.Tensor,
                                            name='cs_' + barra_name + '_neut')

        func_train = partial(getattr(operator, 'cs_ind_neut'), ind=self.barra_train[:, :, 10:])
        self.pset4_train.addPrimitive(func_train, [torch.Tensor], torch.Tensor, name='cs_ind_neut')
        func_predict = partial(getattr(operator, 'cs_ind_neut'), ind=self.barra_predict[:, :, 10:])
        self.pset4_predict.addPrimitive(func_predict, [torch.Tensor], torch.Tensor, name='cs_ind_neut')

        for func_name in operator.variable1_func_list:
            func = getattr(operator, func_name)
            self.pset4_train.addPrimitive(func, [torch.Tensor], torch.Tensor)
            self.pset4_predict.addPrimitive(func, [torch.Tensor], torch.Tensor)

        for func_name in operator.variable2_func_list:
            func = getattr(operator, func_name)
            self.pset4_train.addPrimitive(func, [torch.Tensor, torch.Tensor], torch.Tensor)
            self.pset4_predict.addPrimitive(func, [torch.Tensor, torch.Tensor], torch.Tensor)

        for func_name in operator.variable3_func_list:
            func = getattr(operator, func_name)
            self.pset4_train.addPrimitive(func, [torch.Tensor, torch.Tensor, int], torch.Tensor)
            self.pset4_predict.addPrimitive(func, [torch.Tensor, torch.Tensor, int], torch.Tensor)

        for func_name in operator.on_parameter_func_list:
            func = getattr(operator, func_name)
            self.pset4_train.addPrimitive(func, [torch.Tensor, int], torch.Tensor, name=func_name)
            self.pset4_predict.addPrimitive(func, [torch.Tensor, int], torch.Tensor, name=func_name)

        for part in [0, 1, 2, 3]:
            for method in ['mean', 'std', 'weight_mean', 'prod']:
                func = partial(getattr(operator, 'ts_mask'), part=part, method=method)
                self.pset4_train.addPrimitive(func, [torch.Tensor, torch.Tensor, int], torch.Tensor,
                                              name='ts_mask' + str(part) + '_' + method)
                self.pset4_predict.addPrimitive(func, [torch.Tensor, torch.Tensor, int], torch.Tensor,
                                                name='ts_mask' + str(part) + '_' + method)

        int_values = [int(i) for i in [1, 2, 3, 5, 10, 20, 60]]
        for constant_value in int_values:
            self.pset4_train.addTerminal(constant_value, int)
            self.pset4_predict.addTerminal(constant_value, int)

        constant_function = operator.create_constant_function(2)
        self.pset4_train.addPrimitive(constant_function, [torch.Tensor], int, name='two')
        self.pset4_predict.addPrimitive(constant_function, [torch.Tensor], int, name='two')

        standardize_tool = standardize()
        for func_name in standardize_tool.op_name[:4]:
            func = getattr(standardize_tool, func_name)
            self.pset4_train.addPrimitive(func, [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor)
            self.pset4_predict.addPrimitive(func, [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                                            torch.Tensor)

        for func_name in standardize_tool.op_name[5:]:
            func = getattr(standardize_tool, func_name)
            self.pset4_train.addPrimitive(func, [torch.Tensor, int], torch.Tensor)
            self.pset4_predict.addPrimitive(func, [torch.Tensor, int], torch.Tensor)

        self.pset4_train.addPrimitive(getattr(standardize_tool, standardize_tool.op_name[4]),
                                      [torch.Tensor, torch.Tensor], torch.Tensor, name=standardize_tool.op_name[4])
        self.pset4_predict.addPrimitive(getattr(standardize_tool, standardize_tool.op_name[4]),
                                        [torch.Tensor, torch.Tensor], torch.Tensor, name=standardize_tool.op_name[4])

        creator.create("Linear_factor_train", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset4_train)
        creator.create("Linear_factor_predict", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset4_predict)
        self.toolbox4 = base.Toolbox()
        self.toolbox4.register("expr_train", gp.genHalfAndHalf, pset=self.pset4_train, min_=self.min_tree_depth,
                               max_=self.max_tree_depth)
        self.toolbox4.register("expr_predict", gp.genHalfAndHalf, pset=self.pset4_predict, min_=self.min_tree_depth,
                               max_=self.max_tree_depth)
        self.toolbox4.register("Linear_factor_train", tools.initIterate, creator.Linear_factor_train,
                               self.toolbox4.expr_train)
        self.toolbox4.register("Linear_factor_predict", tools.initIterate, creator.Linear_factor_predict,
                               self.toolbox4.expr_predict)
        self.toolbox4.register("population_train", tools.initRepeat, list, self.toolbox4.Linear_factor_train)
        self.toolbox4.register("population_predict", tools.initRepeat, list, self.toolbox4.Linear_factor_predict)
        self.toolbox4.register("compile_train", gp.compile, pset=self.pset4_train)
        self.toolbox4.register("compile_predict", gp.compile, pset=self.pset4_predict)

    def generate_root(self):
        self.root_population = self.toolbox1.population(n=self.root_size)
        individuals_str = [str(i) for i in self.root_population]
        s = find_unique_elements_with_positions(individuals_str)
        self.root_str = [individuals_str[i] for i in s]
        self.root_code = [self.root_population[i] for i in s]

    def generate_subtree(self):
        self.subtree_population = self.toolbox2.population(n=self.subtree_size)
        individuals_str = [str(i) for i in self.subtree_population]
        s = find_unique_elements_with_positions(individuals_str)

        self.subtree_str = [individuals_str[i] for i in s]
        self.subtree_code = [self.subtree_population[i] for i in s]
        self.subtree_code, self.subtree_str = change_name(self.subtree_code, self.root_code)
    def generate_tree(self):
        self.tree = self.toolbox3.population_train(n=self.tree_size)
        for gen in range(self.num_generations):
            self.offspring = evolve(self.tree, self.toolbox3, cxpb=self.crossover_probability,
                                    mutpb=self.mutation_probability)
            self.offspring_name_changed = change_name_reserve(self.offspring, self.subtree_code)[0]

            truncated_renamed_offspring_formula_code=truncate_formula_name(self.offspring_name_changed,self.pset4_train)
            self.unique_index=find_unique_elements_with_positions([str(s) for s in truncated_renamed_offspring_formula_code])
            self.unique_offspring=[]
            for j in tqdm(self.unique_index):
                fit=self.fitness(truncated_renamed_offspring_formula_code[j])
                self.offspring[j].fitness.values = fit
                self.unique_offspring.append(self.offspring[j])

            self.tree = self.toolbox3.select1(self.unique_offspring, k=len(self.tree))

            selected_tree=change_name_reserve(self.tree, self.subtree_code)[0]
            truncated_selected_tree=truncate_formula_name(selected_tree,self.pset4_train)
            for i in range(len(selected_tree)):
                self.gene[str(truncated_selected_tree[i])]=self.tree[i].fitness.values

        self.trees_code = self.tree
        reserved_index = find_unique_elements_with_positions([str(s) for s in self.tree])
        self.trees_code = [self.trees_code[i] for i in reserved_index]
        self.trees_code,self.trees_str = change_name(self.trees_code, self.subtree_code)

    def out_sample_testing(self):
        self.in_sample_testing_result = {}
        self.out_sample_testing_result = {}
        reserved_index = []
        truncated_trees_code=truncate_formula_name(self.trees_code,self.pset4_train)
        for i in tqdm(range(len(truncated_trees_code))):
            individual = truncated_trees_code[i]
            if self.trees_code[i].fitness.values[0] > config.standard:

                result = self.toolbox4.compile_predict(expr=individual)(*self.predict_list[:])
                cum_return, interval_num = config.fitness_func1(result, self.predict_target,
                                                                self.clean[self.predict_index])
                fitness_1 = cum_return[0] ** (50 / interval_num) - 1
                fitness_2 = cum_return[-1] ** (50 / interval_num) - 1
                fitness_mean = (torch.mean(cum_return) ** (50 / interval_num) - 1).item()
                fitness = abs(max([fitness_1.item() - fitness_mean, fitness_2.item() - fitness_mean]))
                if fitness > config.standard:
                    self.out_sample_testing_result[str(individual)] = fitness
                    self.in_sample_testing_result[str(individual)] = self.trees_code[i].fitness.values[0]
                    reserved_index.append(i)

        self.good_trees = pd.concat([pd.DataFrame(self.in_sample_testing_result, index=['in_sample']).T,
                                     pd.DataFrame(self.out_sample_testing_result, index=['out_sample']).T], axis=1)
        self.good_trees['formula_tree'] = [gp.PrimitiveTree.from_string(i, self.pset4_train) for i in self.good_trees.index]

class Linearation(Generate_Tree):
    def __init__(self):
        super().__init__()

    def linear_test(self, factor, clean=None):
        result, interval_num = config.fitness_func1(factor, self.train_target, clean)
        mean = torch.mean(result)
        if (result[0] > mean) & (result[-1] > mean):
            return 0
        if (result[0] < mean) & (result[-1] > mean):
            return 2
        if (result[0] < mean) & (result[-1] < mean):
            return 3
        if (result[0] > mean) & (result[-1] < mean):
            return 1

    def linearation(self):
        name = []
        self.type_all = []
        s1 = []
        for individual in tqdm(self.good_trees['formula_tree']):
            compiled_func = self.toolbox4.compile_train(expr=individual)
            result = compiled_func(*self.train_list[:])
            type = self.linear_test(result, self.clean[self.train_index])
            self.type_all.append(type)
            op_distance2mean = None
            op_at_neg = None
            for prim in self.pset4_train.primitives[torch.Tensor]:
                if prim.name == 'cs_distance2mean':
                    op_distance2mean = prim
                if prim.name == 'at_neg':
                    op_at_neg = prim

            new_individual = individual
            if type == 0:
                new_individual = gp.PrimitiveTree([op_distance2mean] + [op_at_neg] + individual)
                result = op.cs_distance2mean(-result)
            if type == 1: new_individual = individual
            if type == 2:
                new_individual = gp.PrimitiveTree([op_at_neg] + individual)
                result = -result
            if type == 3:
                new_individual = gp.PrimitiveTree([op_distance2mean] + individual)
                result = op.cs_distance2mean(result)
            cum_return, interval_num = config.fitness_func1(result, self.train_target, self.clean[self.train_index])
            fitness_1 = (cum_return[0] ** (50 / interval_num) - 1).item()
            fitness_mean = (torch.mean(cum_return) ** (50 / interval_num) - 1).item()
            if (fitness_1 - fitness_mean) >= config.standard:
                name.append(new_individual)
                s1.append(fitness_1 - fitness_mean)

        self.lineared_individual_code = {}
        for i in range(len(name)):
            lineared_individual_predict = self.toolbox4.compile_predict(expr=name[i])(*self.predict_list[:])
            cum_return, interval_num = config.fitness_func1(lineared_individual_predict, self.predict_target,
                                                            self.clean[self.predict_index])
            fitness_2 = (cum_return[0] ** (50 / interval_num) - 1).item()
            fitness_mean = (torch.mean(cum_return) ** (50 / interval_num) - 1).item()
            if ((fitness_2 - fitness_mean) >= config.standard):
                self.lineared_individual_code[str(name[i])] = [s1[i], (fitness_2 - fitness_mean), name[i]]

        self.linear_factor = pd.DataFrame(self.lineared_individual_code).T
        self.linear_factor.columns = ['in_sample', 'out_sample', 'formula_tree']

    def correlation_testing(self):
        def filter_dfs():
            add_factor = self.linear_factor['formula_tree']
            exist_factor = pd.read_excel(self.formula_tree_path)['formula_tree']
            exist_factor = [gp.PrimitiveTree.from_string(i, self.pset4_train) for i in exist_factor]
            exist_factor_list = formula_compile(exist_factor, self.toolbox4.compile_train, self.train_list)

            reserved_factor_index = []
            for i, code in tqdm(enumerate(add_factor)):
                f1 = self.toolbox4.compile_train(expr=code)(*self.train_list)
                test_sign = 1
                for f2 in exist_factor_list:
                    if (abs(nanmean(rank_corrwith(f1, f2))) >= 0.6):
                        test_sign = 0
                        break
                if test_sign == 1:
                    reserved_factor_index.append(i)
                    exist_factor_list.append(f1)

            return reserved_factor_index

        self.reserved_factor_index = filter_dfs()
        self.show = self.linear_factor.iloc[self.reserved_factor_index]
        if self.reserved_factor_index is None:
            pass
        else:
            self.new_exist_factor = pd.concat([pd.read_excel(self.formula_tree_path, index_col='Unnamed: 0'),
                                               self.linear_factor.iloc[self.reserved_factor_index]])
            self.new_exist_factor.to_excel(self.formula_tree_path)

    def run(self):
        self.get_basic_data()
        self.generate_toolbox1()
        self.generate_root()
        self.generate_toolbox2()
        self.generate_subtree()
        self.generate_toolbox3()
        self.generate_toolbox4()
        self.generate_tree()
        self.out_sample_testing()
        self.linearation()
        self.correlation_testing()


class factor_return:
    def __init__(self):
        self.min_tree_depth = config.min_tree_depth
        self.max_tree_depth = config.max_tree_depth
        self.train_time = [2016, 2018]
        self.valid_time = [2019, 2021]
        self.predict_time = [2022, 2023]

    def get_bt_list(self):
        self.open, self.close, self.high, self.low = [torch.tensor(datatools.open, dtype=torch.float32, device=device),
                                                      torch.tensor(datatools.close, dtype=torch.float32, device=device),
                                                      torch.tensor(datatools.high, dtype=torch.float32, device=device),
                                                      torch.tensor(datatools.low, dtype=torch.float32, device=device)]
        self.open, self.close, self.high, self.low = [torch.where(i < 1e-5, float('nan'), i) for i in
                                                      [self.open, self.close, self.high, self.low]]
        self.volume = torch.tensor(datatools.get_volume().values, dtype=torch.float32, device=device)
        self.volume = torch.where(self.volume < 1e-5, float('nan'), self.volume)

        date_series = pd.Series(datatools.TradingDate)
        self.index = date_series.index[
            (date_series.dt.year >= self.train_time[0]) & (date_series.dt.year <= self.predict_time[1])]
        self.bt_list = [self.open[self.index], self.close[self.index], self.high[self.index], self.low[self.index],
                        self.volume[self.index]]
        self.bt_list = [torch.where(s < 1e-10, float('nan'), s) for s in self.bt_list]
        pct_change = op.ts_delay(self.close, -5) / op.ts_delay(self.open, -1)
        jump_open = (op.ts_delay(self.open, -1) / self.close > 1.095)
        pct_change = torch.where(torch.isnan(jump_open), float('nan'), pct_change)
        pct_change = torch.where((pct_change == torch.inf) | (pct_change == -torch.inf), float('nan'), pct_change)
        self.target = pct_change[self.index]

    def get_toolbox(self):
        self.pset = gp.PrimitiveSetTyped("MAIN", [torch.Tensor] * len(self.bt_list), torch.Tensor)
        name = ['open', 'close', 'high', 'low', 'volume']
        for i in range(len(self.bt_list)):
            self.pset.renameArguments(**{f'ARG{i}': name[i]})

        dict = torch.load('dict.pt')
        barra = get_barra(range(self.train_time[0], self.predict_time[1] + 1)).to(device)
        operator = op()

        for i, barra_name in enumerate(dict['name'][:10]):
            func_train = partial(getattr(operator, 'cs_barra_neut'), barra_factor=barra[:, :, i])
            self.pset.addPrimitive(func_train, [torch.Tensor], torch.Tensor, name='cs_' + barra_name + '_neut')

        func_train = partial(getattr(operator, 'cs_ind_neut'), ind=barra[:, :, 10:])
        self.pset.addPrimitive(func_train, [torch.Tensor], torch.Tensor, name='cs_ind_neut')

        for func_name in operator.variable1_func_list:
            func = getattr(operator, func_name)
            self.pset.addPrimitive(func, [torch.Tensor], torch.Tensor)

        for func_name in operator.variable2_func_list:
            func = getattr(operator, func_name)
            self.pset.addPrimitive(func, [torch.Tensor, torch.Tensor], torch.Tensor)

        for func_name in operator.variable3_func_list:
            func = getattr(operator, func_name)
            self.pset.addPrimitive(func, [torch.Tensor, torch.Tensor, int], torch.Tensor)

        for func_name in operator.on_parameter_func_list:
            func = getattr(operator, func_name)
            self.pset.addPrimitive(func, [torch.Tensor, int], torch.Tensor, name=func_name)

        for part in [0, 1, 2, 3]:
            for method in ['mean', 'std', 'weight_mean', 'prod']:
                func = partial(getattr(operator, 'ts_mask'), part=part, method=method)
                self.pset.addPrimitive(func, [torch.Tensor, torch.Tensor, int], torch.Tensor,
                                       name='ts_mask' + str(part) + '_' + method)

        int_values = [int(i) for i in [1, 2, 3, 5, 10, 20, 120]]
        for constant_value in int_values:
            self.pset.addTerminal(constant_value, int)

        constant_function = operator.create_constant_function(2)
        self.pset.addPrimitive(constant_function, [torch.Tensor], int, name='two')

        standardize_tool = standardize()
        for func_name in standardize_tool.op_name[:4]:
            func = getattr(standardize_tool, func_name)
            self.pset.addPrimitive(func, [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor)

        for func_name in standardize_tool.op_name[5:]:
            func = getattr(standardize_tool, func_name)
            self.pset.addPrimitive(func, [torch.Tensor, int], torch.Tensor)

        self.pset.addPrimitive(getattr(standardize_tool, standardize_tool.op_name[4]), [torch.Tensor, torch.Tensor],
                               torch.Tensor, name=standardize_tool.op_name[4])

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=self.min_tree_depth,
                              max_=self.max_tree_depth)
        self.toolbox.register("Individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.Individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

    def xcompile(self):
        all_formula = pd.read_excel(config.formula_tree_path, index_col='Unnamed: 0')
        self.part1 = all_formula.dropna(axis=0, how='any')
        self.part2 = all_formula.loc[~all_formula.index.isin(self.part1.index)]

        code_list = self.part2['formula_tree']
        index = datatools.get_pct_change().loc[str(self.train_time[0]):str(self.predict_time[1])].index
        columns = datatools.code
        buydays = index[1::5][:-1]
        selldays = index[5::5]
        df_index = index[::5][:-1]
        self.turn_over = []
        self.train_BT = []
        self.valid_BT = []
        self.predict_BT = []
        self.correlation = []
        for i in tqdm(range(len(code_list))):
            individual = code_list[i]
            compiled_func = self.toolbox.compile(expr=individual)
            result = compiled_func(*self.bt_list[:])
            self.turn_over.append(get_turnover(result).item())

            df = pd.DataFrame(result.cpu(), index=index, columns=columns)
            p = stratified(df.loc[df_index], buydays, selldays, 5)
            p.run()
            self.train_BT.append(p.every_interval_rate.loc[str(self.train_time[0]):str(self.train_time[1])].prod())
            self.valid_BT.append(p.every_interval_rate.loc[str(self.valid_time[0]):str(self.valid_time[1])].prod())
            self.predict_BT.append(
                p.every_interval_rate.loc[str(self.predict_time[0]):str(self.predict_time[1])].prod())
            self.correlation.append(nanmean(rank_corrwith(result[-1000:], self.target[-1000:])).item())

        self.train_BT = pd.concat(self.train_BT, axis=1).T
        self.valid_BT = pd.concat(self.valid_BT, axis=1).T
        self.predict_BT = pd.concat(self.predict_BT, axis=1).T
        self.correlation = pd.DataFrame(self.correlation, index=code_list)
        self.turn_over = pd.DataFrame(self.turn_over, index=code_list)

    def delete(self):
        self.performance = self.predict_BT.apply(lambda x: x.iloc[0] - x.mean(), axis=1)
        self.performance.index = self.part2.index
        self.part2['performance'] = (self.performance + 1) ** (1 / 2) - 1
        self.part2['long_turn_over'] = self.turn_over
        self.part2['IC_rank'] = self.correlation
        self.new = pd.concat([self.part1, self.part2])

    def run(self):
        self.get_bt_list()
        self.get_toolbox()
        self.xcompile()