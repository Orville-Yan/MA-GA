class FIS_Config:
    similarity_threhold = 0.6
    storage_path = r'D:\运行文档\NFE遗传算法项目\MA-GA'

import torch
import os
class RPNbuilder_Config:
    seed_size = 10
    root_size = 10
    branch_size = 10
    trunk_size = 10
    subtree_size = 10
    tree_size = 10

class Root_Config:
    min_depth = 1
    max_depth = 1
    OP_B2B_func_list = ['M_cs_rank', 'M_cs_scale', 'M_cs_zscore']
    OP_BB2B_func_list = ['M_at_div']
    OP_AA2A_func_list = ['D_at_div']
    OP_AF2A_func_list = ['D_ts_pctchg','D_ts_norm']
    default_lookback = [2, 3, 5, 10, 20]

class Branch_Config:
    min_depth = 1
    max_depth = 1
    default_lookback = [2, 3, 5, 10, 20]

class Seed_Config:
    min_depth = 1
    max_depth = 1
    OP_AF2A_func_list = ['D_ts_max', 'D_ts_min','D_ts_delay', 'D_ts_delta', 'D_ts_mean']
    OP_AA2A_func_list = ['D_at_mean']
    default_lookback = [2, 3, 5,  10, 20]
    OP_BF2B_func_list = ['M_ts_delay', 'M_ts_mean_left_neighbor',
                                  'M_ts_mean_mid_neighbor', 'M_ts_mean_right_neighbor']

class Trunk_Config:
    default_lookback = [2, 3, 5, 10, 20]
    default_edge = [0.05, 0.1]
    min_depth = 1
    max_depth = 3

class Subtree_Config:
    min_depth = 1
    max_depth = 1
    OP_B2A_func_list = ['D_Minute_std', 'D_Minute_mean', 'D_Minute_trend']
    OP_BB2A_func_list  = ['D_Minute_corr', 'D_Minute_weight_mean']
    OP_BD2A_func_list = ['D_Minute_area_mean', 'D_Minute_area_std', 'D_Minute_area_sum', 'D_Minute_area_prod']
    OP_BBD2A_func_list = ['D_Minute_area_weight_mean', 'D_Minute_area_corr', 'D_Minute_area_rankcorr',
                                   'D_Minute_area_bifurcate_mean', 'D_Minute_area_bifurcate_std']
class Tree_Config:
    min_depth = 1
    max_depth = 1
    OP_AF2A_func_list = ['D_ts_mean', 'D_ts_harmonic_mean', 'D_ts_std']
    default_lookback = [2, 3, 5, 10, 20]

class BackTest_Config:
    # FactorTest parameters
    bins_num = 5
    period_num = 252
    default_year = [2016]

class Data_Config:
    DEVICE = 'cpu'
    MINUTE_LEN = 242
    DATA_PATH = "../Data"
    PARQUET_Minute_PATH = os.path.join(DATA_PATH, "Minute")
    PARQUET_Daily_PATH = os.path.join(DATA_PATH, "Daily")
    PARQUET_BARRA_PATH = os.path.join(DATA_PATH, "Barra.pt")
    PARQUET_DICT_PATH = os.path.join(DATA_PATH, "dict.pt")
    MMAP_PATH = os.path.join(DATA_PATH, "Mmap")
    MMAP_Minute_PATH = os.path.join(MMAP_PATH, "Minute")
    MMAP_Daily_PATH = os.path.join(MMAP_PATH, "Daily")
    MMAP_BARRA_PATH = os.path.join(MMAP_PATH, "Barra")
    MUTUAL_STOCK_CODES_PATH = os.path.join(DATA_PATH, "MutualStockCodes.parquet")

