op_info={
    'D_at_abs': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '计算输入张量的绝对值'
        }
    },
    'D_cs_rank': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '计算输入张量的截面分位数，忽略 NaN 值'
        }
    },
    'D_cs_scale': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '对输入张量进行截面标准化，标准化到 [0, 1] 区间'
        }
    },
    'D_cs_zscore': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '对输入张量进行截面 z-score 标准化'
        }
    },
    'D_cs_harmonic_mean': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '计算输入张量的截面调和平均值，忽略 NaN 和 0 值'
        }
    },
    'D_cs_demean': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '对输入张量进行截面去均值处理，并取绝对值'
        }
    },
    'D_cs_winsor': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'limit': 'list, 默认为 [0.05, 0.95], 表示尾部磨平的分位数范围'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '对输入张量进行尾部磨平处理，将分位数小于 limit[0] 或大于 limit[1] 的部分替换为对应分位数的值'
        }
    },
    'D_cs_norm_spread': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'y': 'shape=(day_len, num_stock), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '计算 (x - y) / (|x| + |y|)'
        }
    },
    'D_cs_cut': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'y': 'shape=(day_len, num_stock), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '根据 x 的符号对 y 进行截断处理'
        }
    },
    'D_cs_regress_res': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'y': 'shape=(day_len, num_stock), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '对 x 和 y 进行截面回归，返回残差'
        }
    },
    'D_at_add': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'y': 'shape=(day_len, num_stock), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '对 x 和 y 进行逐元素相加'
        }
    },
    'D_at_div': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'y': 'shape=(day_len, num_stock), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '对 x 和 y 进行逐元素相除，y 为 0 时返回 NaN'
        }
    },
    'D_at_sub': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'y': 'shape=(day_len, num_stock), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '对 x 和 y 进行逐元素相减'
        }
    },
    'D_at_prod': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'y': 'shape=(day_len, num_stock), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '对 x 和 y 进行逐元素相除，y 为 0 或 NaN 时返回 NaN'
        }
    },
    'D_at_mean': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'y': 'shape=(day_len, num_stock), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '计算 x 和 y 的逐元素均值'
        }
    },
    'D_cs_edge_flip': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'thresh': 'float, 阈值'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '根据阈值 thresh 对 x 的边缘部分进行翻转处理'
        }
    },
    'D_ts_corr': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'y': 'shape=(day_len, num_stock), 输入的张量',
            'd': 'int, 回溯的天数'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '计算 x 和 y 在回溯 d 天内的时序相关性'
        }
    },
    'D_ts_rankcorr': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'y': 'shape=(day_len, num_stock), 输入的张量',
            'd': 'int, 回溯的天数'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '计算 x 和 y 在回溯 d 天内的时序秩相关性'
        }
    },
    'D_ts_regress_res': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'y': 'shape=(day_len, num_stock), 输入的张量',
            'lookback': 'int, 回溯的天数'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '对 x 和 y 进行时序回归，返回残差'
        }
    },
    'D_ts_weight_mean': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'y': 'shape=(day_len, num_stock), 输入的张量',
            'lookback': 'int, 回溯的天数'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '以 y 为权重，计算 x 在回溯 lookback 天内的加权平均值'
        }
    },
    'D_ts_max': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'lookback': 'int, 回溯的天数'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '计算 x 在回溯 lookback 天内的最大值'
        }
    },
    'D_ts_min': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'lookback': 'int, 回溯的天数'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '计算 x 在回溯 lookback 天内的最小值'
        }
    },
    'D_ts_delay': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'd': 'int, 延迟的天数'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '对 x 进行 d 天的延迟处理'
        }
    },
    'D_ts_delta': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'd': 'int, 延迟的天数'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '计算 x 与 d 天前的差值'
        }
    },
    'D_ts_pctchg': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'lookback': 'int, 回溯的天数'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '计算 x 在回溯 lookback 天内的涨跌幅'
        }
    },
    'D_ts_mean': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'lookback': 'int, 回溯的天数'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '计算 x 在回溯 lookback 天内的均值'
        }
    },
    'D_ts_harmonic_mean': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'lookback': 'int, 回溯的天数'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '计算 x 在回溯 lookback 天内的调和平均值'
        }
    },
    'D_ts_std': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'lookback': 'int, 回溯的天数'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '计算 x 在回溯 lookback 天内的标准差'
        }
    },
    'D_ts_to_max': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'lookback': 'int, 回溯的天数'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '计算 x 与回溯 lookback 天内的最大值的比值'
        }
    },
    'D_ts_to_min': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'lookback': 'int, 回溯的天数'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '计算 x 与回溯 lookback 天内的最小值的比值'
        }
    },
    'D_ts_to_mean': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'lookback': 'int, 回溯的天数'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '计算 x 与回溯 lookback 天内的均值的比值'
        }
    },
    'D_ts_max_to_min': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'lookback': 'int, 回溯的天数'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '计算 x 与回溯 lookback 天内的最大值与最小值的差值的比值'
        }
    },
    'D_ts_maxmin_norm': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'lookback': 'int, 回溯的天数'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '对 x 进行回溯 lookback 天内的最大最小值归一化'
        }
    },
    'D_ts_norm': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'lookback': 'int, 回溯的天数'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '对 x 进行回溯 lookback 天内的 z-score 标准化'
        }
    },
    'D_ts_detrend': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'lookback': 'int, 回溯的天数'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '去除 x 在回溯 lookback 天内的趋势'
        }
    },
    'D_ts_mask_mean': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'mask': 'shape=(day_len, num_stock, rolling_days), 掩码张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '对 mask 为 True 的部分计算 x 的均值'
        }
    },
    'D_ts_mask_std': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'mask': 'shape=(day_len, num_stock, rolling_days), 掩码张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '对 mask 为 True 的部分计算 x 的标准差'
        }
    },
    'D_ts_mask_sum': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'mask': 'shape=(day_len, num_stock, rolling_days), 掩码张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '对 mask 为 True 的部分计算 x 的和'
        }
    },
    'D_ts_mask_prod': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'mask': 'shape=(day_len, num_stock, rolling_days), 掩码张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '对 mask 为 True 的部分计算 x 的乘积'
        }
    },
    'D_Minute_area_mean': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'mask': 'shape=(day_len, num_stock, rolling_days), 掩码张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '对 mask 为 True 的部分计算 x 的均值'
        }
    },
    'D_Minute_area_std': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'mask': 'shape=(day_len, num_stock, rolling_days), 掩码张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '对 mask 为 True 的部分计算 x 的标准差'
        }
    },
    'D_Minute_area_sum': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'mask': 'shape=(day_len, num_stock, rolling_days), 掩码张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '对 mask 为 True 的部分计算 x 的和'
        }
    },
    'D_Minute_area_prod': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'mask': 'shape=(day_len, num_stock, rolling_days), 掩码张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '对 mask 为 True 的部分计算 x 的乘积'
        }
    },
    'D_Minute_std': {
        'input_parameters': {
            'm_tensor': 'shape=(day_len, num_stock, minute_len), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '计算日内数据的标准差'
        }
    },
    'D_Minute_mean': {
        'input_parameters': {
            'm_tensor': 'shape=(day_len, num_stock, minute_len), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '计算日内数据的均值'
        }
    },
    'D_Minute_trend': {
        'input_parameters': {
            'm_tensor': 'shape=(day_len, num_stock, minute_len), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '计算日内数据的变化趋势'
        }
    },
    'D_Minute_area_weight_mean': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'weight': 'shape=(day_len, num_stock, rolling_days), 权重张量',
            'mask': 'shape=(day_len, num_stock, rolling_days), 掩码张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '对 mask 为 True 的部分，以 weight 为权重计算 x 的加权均值'
        }
    },
    'D_Minute_area_corr': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'y': 'shape=(day_len, num_stock), 输入的张量',
            'mask': 'shape=(day_len, num_stock, rolling_days), 掩码张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '对 mask 为 True 的部分计算 x 和 y 的相关性'
        }
    },
    'D_Minute_area_rankcorr': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'y': 'shape=(day_len, num_stock), 输入的张量',
            'mask': 'shape=(day_len, num_stock, rolling_days), 掩码张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '对 mask 为 True 的部分计算 x 和 y 的秩相关性'
        }
    },
    'D_Minute_area_bifurcate_mean': {
        'input_parameters': {
            'm_tensor_x': 'shape=(day_len, num_stock, minute_len), 输入的张量',
            'm_tensor_y': 'shape=(day_len, num_stock, minute_len), 输入的张量',
            'mask': 'shape=(day_len, num_stock, rolling_days), 掩码张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '根据 m_tensor_y 的均值将 m_tensor_x 分为两部分，分别计算均值并相减'
        }
    },
    'D_Minute_area_bifurcate_std': {
        'input_parameters': {
            'm_tensor_x': 'shape=(day_len, num_stock, minute_len), 输入的张量',
            'm_tensor_y': 'shape=(day_len, num_stock, minute_len), 输入的张量',
            'mask': 'shape=(day_len, num_stock, rolling_days), 掩码张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '根据 m_tensor_y 的均值将 m_tensor_x 分为两部分，分别计算标准差并相减'
        }
    },
    'D_Minute_corr': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'y': 'shape=(day_len, num_stock), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '计算 x 和 y 的相关性'
        }
    },
    'D_Minute_weight_mean': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'weight': 'shape=(day_len, num_stock), 权重张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '以 weight 为权重计算 x 的加权均值'
        }
    },
    'D_Minute_abnormal_point_count': {
        'input_parameters': {
            'mask': 'shape=(day_len, num_stock, rolling_days), 掩码张量'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock)',
            'description': '计算 mask 中异常点的数量'
        }
    },

    'M_ignore_wobble': {
        'input_parameters': {
            'M_tensor': 'shape=(num_stock, day_len, minute_len), 输入的张量',
            'window_size': 'int, 默认为5, 表示忽略的窗口大小（分钟）'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '将开盘前 window_size 分钟和收盘前 window_size 分钟的数据设置为 NaN'
        }
    },
    'M_cs_zscore': {
        'input_parameters': {
            'M_tensor': 'shape=(num_stock, day_len, minute_len), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '对每个股票的每日数据进行 z-score 标准化'
        }
    },
    'M_cs_rank': {
        'input_parameters': {
            'M_tensor': 'shape=(num_stock, day_len, minute_len), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '对每个股票的每日数据进行排名标准化，范围为 [0, 1]'
        }
    },
    'M_cs_scale': {
        'input_parameters': {
            'M_tensor': 'shape=(num_stock, day_len, minute_len), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '对每个股票的每日数据进行最大值最小值标准化，范围为 [0, 1]'
        }
    },
    'M_cs_demean': {
        'input_parameters': {
            'M_tensor': 'shape=(num_stock, day_len, minute_len), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '对每个股票的每日数据减去其均值'
        }
    },
    'M_cs_winsor': {
        'input_parameters': {
            'M_tensor': 'shape=(num_stock, day_len, minute_len), 输入的张量',
            'lower_percentile': 'float, 默认为0.05, 下百分位数',
            'upper_percentile': 'float, 默认为0.95, 上百分位数'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '对每个股票的每日数据进行 Winsor 处理，将超出百分位数的值替换为对应百分位数的值'
        }
    },
    'M_at_abs': {
        'input_parameters': {
            'M_tensor': 'shape=(num_stock, day_len, minute_len), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '计算输入张量的绝对值'
        }
    },
    'M_ts_delay': {
        'input_parameters': {
            'x': 'shape=(num_stock, day_len, minute_len), 输入的张量',
            'd': 'int, 延迟的天数'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '对输入张量进行 d 天的延迟处理'
        }
    },
    'M_ts_pctchg': {
        'input_parameters': {
            'x': 'shape=(num_stock, day_len, minute_len), 输入的张量',
            'lookback': 'int, 回溯的天数'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '计算输入张量在回溯 lookback 天内的涨跌幅'
        }
    },
    'M_at_add': {
        'input_parameters': {
            'x': 'shape=(num_stock, day_len, minute_len), 输入的张量',
            'y': 'shape=(num_stock, day_len, minute_len), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '对 x 和 y 进行逐元素相加'
        }
    },
    'M_at_sub': {
        'input_parameters': {
            'x': 'shape=(num_stock, day_len, minute_len), 输入的张量',
            'y': 'shape=(num_stock, day_len, minute_len), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '对 x 和 y 进行逐元素相减'
        }
    },
    'M_at_div': {
        'input_parameters': {
            'x': 'shape=(num_stock, day_len, minute_len), 输入的张量',
            'y': 'shape=(num_stock, day_len, minute_len), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '对 x 和 y 进行逐元素相除，y 为 0 时返回 NaN'
        }
    },
    'M_at_sign': {
        'input_parameters': {
            'x': 'shape=(num_stock, day_len, minute_len), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '计算输入张量的符号，生成掩码'
        }
    },
    'M_cs_cut': {
        'input_parameters': {
            'x': 'shape=(num_stock, day_len, minute_len), 输入的张量',
            'y': 'shape=(num_stock, day_len, minute_len), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '基于 x 的均值调整符号，并将调整后的符号应用于 y'
        }
    },
    'M_cs_umr': {
        'input_parameters': {
            'x': 'shape=(num_stock, day_len, minute_len), 输入的张量',
            'y': 'shape=(num_stock, day_len, minute_len), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '计算 (x - mean(y)) * y'
        }
    },
    'M_at_prod': {
        'input_parameters': {
            'd_tensor_x': 'shape=(num_stock, day_len, minute_len), 输入的张量',
            'd_tensor_y': 'shape=(num_stock, day_len, minute_len), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '对 d_tensor_x 和 d_tensor_y 进行逐元素相除，d_tensor_y 为 0 或 NaN 时返回 NaN'
        }
    },
    'M_cs_norm_spread': {
        'input_parameters': {
            'x': 'shape=(num_stock, day_len, minute_len), 输入的张量',
            'y': 'shape=(num_stock, day_len, minute_len), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '计算 (x - y) / (|x| + |y|)'
        }
    },
    'M_toD_standard': {
        'input_parameters': {
            'M_tensor': 'shape=(num_stock, day_len, minute_len), 输入的张量',
            'D_tensor': 'shape=(day_len, num_stock), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '将 M_tensor 除以 D_tensor 的转置并扩展维度后的结果'
        }
    },
    'M_cs_edge_flip': {
        'input_parameters': {
            'M_tensor': 'shape=(num_stock, day_len, minute_len), 输入的张量',
            'thresh': 'float, 阈值'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '根据阈值 thresh 对 M_tensor 的边缘部分进行翻转处理'
        }
    },
    'M_ts_delta': {
        'input_parameters': {
            'm_tensor': 'shape=(num_stock, day_len, minute_len), 输入的张量',
            'lookback': 'int, 回溯的天数'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '计算 m_tensor 与回溯 lookback 天前的差值'
        }
    },
    'M_ts_mean_left_neighbor': {
        'input_parameters': {
            'm_tensor': 'shape=(num_stock, day_len, minute_len), 输入的张量',
            'neighbor_range': 'int, 范围'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '计算 m_tensor 左侧一定范围的均值'
        }
    },
    'M_ts_mean_mid_neighbor': {
        'input_parameters': {
            'm_tensor': 'shape=(num_stock, day_len, minute_len), 输入的张量',
            'neighbor_range': 'int, 范围'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '计算 m_tensor 中间一定范围的均值'
        }
    },
    'M_ts_mean_right_neighbor': {
        'input_parameters': {
            'm_tensor': 'shape=(num_stock, day_len, minute_len), 输入的张量',
            'neighbor_range': 'int, 范围'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '计算 m_tensor 右侧一定范围的均值'
        }
    },
    'M_ts_std_left_neighbor': {
        'input_parameters': {
            'm_tensor': 'shape=(num_stock, day_len, minute_len), 输入的张量',
            'neighbor_range': 'int, 范围'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '计算 m_tensor 左侧一定范围的标准差'
        }
    },
    'M_ts_std_mid_neighbor': {
        'input_parameters': {
            'm_tensor': 'shape=(num_stock, day_len, minute_len), 输入的张量',
            'neighbor_range': 'int, 范围'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '计算 m_tensor 中间一定范围的标准差'
        }
    },
    'M_ts_std_right_neighbor': {
        'input_parameters': {
            'm_tensor': 'shape=(num_stock, day_len, minute_len), 输入的张量',
            'neighbor_range': 'int, 范围'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '计算 m_tensor 右侧一定范围的标准差'
        }
    },
    'M_ts_product_left_neighbor': {
        'input_parameters': {
            'm_tensor': 'shape=(num_stock, day_len, minute_len), 输入的张量',
            'neighbor_range': 'int, 范围'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '计算 m_tensor 左侧一定范围的乘积'
        }
    },
    'M_ts_product_mid_neighbor': {
        'input_parameters': {
            'm_tensor': 'shape=(num_stock, day_len, minute_len), 输入的张量',
            'neighbor_range': 'int, 范围'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '计算 m_tensor 中间一定范围的乘积'
        }
    },
    'M_ts_product_right_neighbor': {
        'input_parameters': {
            'm_tensor': 'shape=(num_stock, day_len, minute_len), 输入的张量',
            'neighbor_range': 'int, 范围'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '计算 m_tensor 右侧一定范围的乘积'
        }
    },
    

    'Dmask_min': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'lookback': 'int, 回溯的天数'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock, lookback)',
            'description': '对 x 进行 unfold 操作，回溯 lookback 天的数据，取最小的 1/4 天作为掩码'
        }
    },
    'Dmask_max': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'lookback': 'int, 回溯的天数'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock, lookback)',
            'description': '对 x 进行 unfold 操作，回溯 lookback 天的数据，取最大的 1/4 天作为掩码'
        }
    },
    'Dmask_middle': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'lookback': 'int, 回溯的天数'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock, lookback)',
            'description': '对 x 进行 unfold 操作，回溯 lookback 天的数据，取中间的 1/2 天作为掩码'
        }
    },
    'Dmask_mean_plus_std': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'lookback': 'int, 回溯的天数'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock, lookback)',
            'description': '对 x 进行 unfold 操作，回溯 lookback 天的数据，进行标准化处理，取大于均值 + 标准差的部分作为掩码'
        }
    },
    'Dmask_mean_sub_std': {
        'input_parameters': {
            'x': 'shape=(day_len, num_stock), 输入的张量',
            'lookback': 'int, 回溯的天数'
        },
        'output': {
            'tensor': 'shape=(day_len, num_stock, lookback)',
            'description': '对 x 进行 unfold 操作，回溯 lookback 天的数据，进行标准化处理，取小于均值 - 标准差的部分作为掩码'
        }
    },


    'Mmask_min': {
        'input_parameters': {
            'x': 'shape=(num_stock, day_len, minute_len), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '返回日内数据的最小 1/4 部分作为掩码'
        }
    },
    'Mmask_max': {
        'input_parameters': {
            'x': 'shape=(num_stock, day_len, minute_len), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '返回日内数据的最大 1/4 部分作为掩码'
        }
    },
    'Mmask_middle': {
        'input_parameters': {
            'x': 'shape=(num_stock, day_len, minute_len), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '返回日内数据的中间 1/2 部分作为掩码'
        }
    },
    'Mmask_min_to_max': {
        'input_parameters': {
            'x': 'shape=(num_stock, day_len, minute_len), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '返回日内数据的最小值和最大值之间的部分作为掩码'
        }
    },
    'Mmask_mean_plus_std': {
        'input_parameters': {
            'x': 'shape=(num_stock, day_len, minute_len), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '返回日内数据标准化后大于均值 + 标准差的部分作为掩码'
        }
    },
    'Mmask_mean_sub_std': {
        'input_parameters': {
            'x': 'shape=(num_stock, day_len, minute_len), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '返回日内数据标准化后小于均值 - 标准差的部分作为掩码'
        }
    },
    'Mmask_1h_after_open': {
        'input_parameters': {
            'x': 'shape=(num_stock, day_len, minute_len), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, 60)',
            'description': '返回开盘后第 1 个小时的数据'
        }
    },
    'Mmask_1h_before_close': {
        'input_parameters': {
            'x': 'shape=(num_stock, day_len, minute_len), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, 60)',
            'description': '返回收盘前第 1 个小时的数据'
        }
    },
    'Mmask_2h_middle': {
        'input_parameters': {
            'x': 'shape=(num_stock, day_len, minute_len), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, 120)',
            'description': '返回中间 2 个小时的数据'
        }
    },
    'Mmask_morning': {
        'input_parameters': {
            'x': 'shape=(num_stock, day_len, minute_len), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, 120)',
            'description': '返回早上 2 个小时的数据'
        }
    },
    'Mmask_afternoon': {
        'input_parameters': {
            'x': 'shape=(num_stock, day_len, minute_len), 输入的张量'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, 120)',
            'description': '返回下午 2 个小时的数据'
        }
    },
    'Mmask_day_plus': {
        'input_parameters': {
            'm_tensor': 'shape=(num_stock, day_len, minute_len), 输入的张量',
            'd_tensor': 'shape=(day_len, num_stock), 输入的日频数据'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '返回大于日频数据的部分作为掩码'
        }
    },
    'Mmask_day_sub': {
        'input_parameters': {
            'm_tensor': 'shape=(num_stock, day_len, minute_len), 输入的张量',
            'd_tensor': 'shape=(day_len, num_stock), 输入的日频数据'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '返回小于日频数据的部分作为掩码'
        }
    },
    'Mmask_rolling_plus': {
        'input_parameters': {
            'm_tensor': 'shape=(num_stock, day_len, minute_len), 输入的张量',
            'lookback': 'int, 回溯的天数'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '以日内数据最大 1/4 部分的均值作为日较大值，返回大于 lookback 期内最大日较大值的部分作为掩码'
        }
    },
    'Mmask_rolling_sub': {
        'input_parameters': {
            'm_tensor': 'shape=(num_stock, day_len, minute_len), 输入的张量',
            'lookback': 'int, 回溯的天数'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '以日内数据最小 1/4 部分的均值作为日较大值，返回小于 lookback 期内最小日较大值的部分作为掩码'
        }
    },
    'Mmask_and': {
        'input_parameters': {
            'm_mask_x': 'shape=(num_stock, day_len, minute_len), 输入的掩码张量',
            'm_mask_y': 'shape=(num_stock, day_len, minute_len), 输入的掩码张量'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '对两个掩码张量进行逻辑与运算'
        }
    },
    'Mmask_or': {
        'input_parameters': {
            'm_mask_x': 'shape=(num_stock, day_len, minute_len), 输入的掩码张量',
            'm_mask_y': 'shape=(num_stock, day_len, minute_len), 输入的掩码张量'
        },
        'output': {
            'tensor': 'shape=(num_stock, day_len, minute_len)',
            'description': '对两个掩码张量进行逻辑或运算'
        }
    }
}
