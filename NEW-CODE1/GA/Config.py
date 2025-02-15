import torch
class RPNbuilder_Config:
    SEED_SIZE = 10
    ROOT_SIZE = 10
    BRANCH_SIZE = 10
    TRUNK_SIZE = 10
    SUBTREE_SIZE = 10
    TREE_SIZE = 10
    DEFAULT_YEAR = [2016]
    DEFAULT_POPULATION = 10
    DEFAULT_LOOKBACK = [2, 3, 5, 10, 20]
    DEFAULT_EDGE = [0.05, 0.1]
    MIN_DEPTH = 1
    MAX_DEPTH = 10

class Root_Config:
    DEFAULT_POPULATION = 10
    DEFAULT_INT = [2, 3, 5, 10, 20]
    MIN_DEPTH = 1
    MAX_DEPTH = 1

class Branch_Config:
    DEFAULT_POPULATION = 10
    DEFAULT_LOOKBACK = [2, 3, 5, 10, 20]
    MIN_DEPTH = 1
    MAX_DEPTH = 1

class Seed_Config:
    DEFAULT_POPULATION = 10
    DEFAULT_LOOKBACK = [2, 3, 5, 10, 20]
    MIN_DEPTH = 1
    MAX_DEPTH = 1

class Trunk_Config:
    DEFAULT_POPULATION = 10
    DEFAULT_LOOKBACK = [2, 3, 5, 10, 20]
    DEFAULT_EDGE = [0.05, 0.1]
    MIN_DEPTH = 1
    MAX_DEPTH = 1

class Subtree_Config:
    DEFAULT_POPULATION = 10
    MIN_DEPTH = 1
    MAX_DEPTH = 1

class Tree_Config:
    DEFAULT_POPULATION = 10
    MIN_DEPTH = 1
    MAX_DEPTH = 1

class BackTest_Config:
    # FactorTest parameters
    bins_num = 5
    period_num = 252
    default_year = [2016]

class Data_tools_Config:
    DAILY_DATA_PATH = "../Data/DailyData"
    MUTUAL_STOCK_CODES_PATH = "../Data/MutualStockCodes.parquet"
    MINUTE_DATA_PATH = "../../Data/MinuteData"
    DEVICE = 'cpu'
    MINUTE_LEN = 242
    COLS = ["open", "high", "low", "close", "volume"]

class DataReader_Config:
    # Paths
    DailyDataPath = "../Data/DailyData"
    MinuteDataPath = "../Data/MinuteData"
    BarraPath = "../Data/barra.pt"
    DictPath = "../Data/dict.pt"
    
    # Device
    device = 'cpu'

    
    # MmapReader settings
    years = range(2017, 2018)
    output_daily = '..'
    output_minute = '..'

class GA_tools_Config:
    # Chaotic map parameters
    chebyshev_a = 4
    circle_a = 0.5
    circle_b = 2.2
    iterative_a = 0.7
    logistic_a = 4
    piecewise_d = 0.3
    sine_a = 4
    singer_a = 1.07
    tent_a = 0.4
    spm_eta = 0.4
    spm_mu = 0.3
    spm_r = torch.rand(1)
    tent_logistic_cosine_r = 0.7
    sine_tent_cosine_r = 0.7
    logistic_sine_cosine_r = 0.7
    cubic_a = 2.595
    logistic_tent_r = 0.3
    bernoulli_a = 0.4 