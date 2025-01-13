data_info={
    'D_O':{
        'shape':'(day_len,num_stock)',
        'description':'当日开盘价'
    },
    'D_C':{
        'shape':'(day_len,num_stock)',
        'description':'当日收盘价'
    },
    'D_H':{
        'shape':'(day_len,num_stock)',
        'description':'当日最高价'
    },
    'D_L':{
        'shape':'(day_len,num_stock)',
        'description':'当日最低价'
    },
    'D_V':{
        'shape':'(day_len,num_stock)',
        'description':'当日成交额'
    },
    'M_O':{
        'shape':'(day_len,num_stock,minute_len)',
        'description':'日内每一分钟的开盘价'
    },
    'M_C':{
        'shape':'(day_len,num_stock,minute_len)',
        'description':'日内每一分钟的收盘价'
    },
    'M_H':{
        'shape':'(day_len,num_stock,minute_len)',
        'description':'日内每一分钟的最高价'
    },
    'M_L':{
        'shape':'(day_len,num_stock,minute_len)',
        'description':'日内每一分钟的最低价'
    },
    'M_V':{
        'shape':'(day_len,num_stock,minute_len)',
        'description':'日内每一分钟的成交额'
    },
}

VariableType_info={
    'TypeA':{'shape':'(day_len, num_stock)的二维tensor',
             'name':'day_OHLCV',
             '性质':'日频的K线数据'
             },
    'TypeB':{'shape':'(num_stock, day_len, minute_len=240)的三维tensor',
             'name':'minute_OHLCV',
             '性质':'分钟频的K线数据'
             },
    'TypeC':{'shape':'(day_len, num_stock, rolling_day)的三维tensor',
             'name':'day_mask',
             '性质':'日频掩码'
             },
    'TypeD':{'shape':'(num_stock, day_len, minute_len=240)的三维tensor',
             'name':'minute_mask',
             '性质':'分钟频掩码'
             },
    'TypeE':{'shape':'(day_len, num_stock, industry_num=31)的三维tensor',
             'name':'industry',
             '性质':'按行业划分的日频数据'
             },
    'TypeF':{'shape':'一维tensor',
             'name':'time_int',
             '性质':'时间戳'
             },
    'TypeG':{'shape':'float变量',
             'name':'threshold',
             '性质':'反转阈值'
             }
}

Interface_Protocol_Group = {
    '原则':'将算子按照输入和输出的变量类型以及混频同频等格式作区分，总共有五个层级，分别是纲目科属种。纲有两类，表示算子的使用领域；目有四类，表示算子的输出类型；科有两类，表示算子使用的变量类型频度；属的类别较多，表示算子输入的变量组合。种对应到单个算子。这种分类是一种无交叉的分类，每个算子只可能属于一个类别',
    '具体划分':{
    '纲':{
        '通用纲':'该纲中的算子是用途广泛的数据操作动作，可以用于几乎所有变量类型，一般用于生成专用纲算子中的传递变量；Others.py中的算子均为通用纲算子',
        '专用纲':'该纲中的算子专门用于生成TypeABCD四种变量类型，专用纲算子均在ToA.py，ToB.py，ToC.py，ToD.py中。它包含的算子类型多样，需要继续用目科属层级进行划分'
        },
    '目':{
        'A目':'输出类型为TypeA',
        'B目':'输出类型为TypeB',
        'C目':'输出类型为TypeC',
        'D目':'输出类型为TypeD'
        },
    '科':{
        '同频科':'输入的 变量类型组合和输出的变量类型 全部都属于日频或者分钟频中的同一个频度',
        '混频科':'输入的 变量类型组合和输出的变量类型 既包含日频，也包含分钟频'
        },
    '属':{
        '属的划分依据是输入的变量类型组合，如BBD属表示输入的变量类型有三个，分别是TypeB,TypeB 和 TypeD；A属表示输入的变量类型只有TypeA'
    },
    },
}
    
Operator_classification_info={
    
}

Abstraction_Action_Group={
    'name':{'action;抽象动作分类:根据算子是否改变算子特定维度中的rank进行分类'
            },
    'rule':{
    '共轭:conjugate,不改变rank，如M_cs_scale',
    '扭曲:twist,产生了易懂直观的结构性变化，如edge_flip',
    '混沌:chaos,四种基本运算及其延伸与组合',
    '自适应:self-adjust,在特定维度的rank变化受自身特性影响，如D_ts_to_max'
    '剥离:strip,回归类的算子，剥离出数据的某些特征'
    '描述:describe,包含统计操作，如求均值，方差等'
    '提纯:purify，包含mask掩码操作'
    },
    'example':{ }
}

op_info = {
    'D_at_abs': {
        'description': '计算输入张量的绝对值',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'A属'
            }
        }
    },
    'D_cs_rank': {
        'description': '计算输入张量的截面分位数，忽略 NaN 值',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'A属'
            }
        }
    },
    'D_cs_scale': {
        'description': '对输入张量进行截面标准化，标准化到 [0, 1] 区间',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'A属'
            }
        }
    },
    'D_cs_zscore': {
        'description': '对输入张量进行截面 z-score 标准化',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'A属'
            }
        }
    },
    'D_cs_harmonic_mean': {
        'description': '计算输入张量的截面调和平均值，忽略 NaN 和 0 值',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'A属'
            }
        }
    },
    'D_cs_demean': {
        'description': '对输入张量进行截面去均值处理，并取绝对值',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'A属'
            }
        }
    },
    'D_cs_winsor': {
        'description': '对输入张量进行尾部磨平处理，将分位数小于 limit[0] 或大于 limit[1] 的部分替换为对应分位数的值',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'A属'
            }
        }
    },
    'D_cs_norm_spread': {
        'description': '计算 (x - y) / (|x| + |y|)',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AA属'
            }
        }
    },
    'D_cs_cut': {
        'description': '根据 x 的符号对 y 进行截断处理',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AA属'
            }
        }
    },
    'D_cs_regress_res': {
        'description': '对 x 和 y 进行截面回归，返回残差',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AA属'
            }
        }
    },
    'D_at_add': {
        'description': '对 x 和 y 进行逐元素相加',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AA属'
            }
        }
    },
    'D_at_div': {
        'description': '对 x 和 y 进行逐元素相除，y 为 0 时返回 NaN',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AA属'
            }
        }
    },
    'D_at_sub': {
        'description': '对 x 和 y 进行逐元素相减',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AA属'
            }
        }
    },
    'D_at_prod': {
        'description': '对 x 和 y 进行逐元素相除，y 为 0 或 NaN 时返回 NaN',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AA属'
            }
        }
    },
    'D_at_mean': {
        'description': '计算 x 和 y 的逐元素均值',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AA属'
            }
        }
    },
    'D_cs_edge_flip': {
        'description': '根据阈值 thresh 对 x 的边缘部分进行翻转处理',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AG属'
            }
        }
    },
    'D_ts_corr': {
        'description': '计算 x 和 y 在回溯 d 天内的时序相关性',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AAF属'
            }
        }
    },
    'D_ts_rankcorr': {
        'description': '计算 x 和 y 在回溯 d 天内的时序秩相关性',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AAF属'
            }
        }
    },
    'D_ts_regress_res': {
        'description': '对 x 和 y 进行时序回归，返回残差',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AAF属'
            }
        }
    },
    'D_ts_weight_mean': {
        'description': '以 y 为权重，计算 x 在回溯 lookback 天内的加权平均值',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AAF属'
            }
        }
    },
    'D_ts_max': {
        'description': '计算 x 在回溯 lookback 天内的最大值',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            }
        }
    },
    'D_ts_min': {
        'description': '计算 x 在回溯 lookback 天内的最小值',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            }
        }
    },
    'D_ts_delay': {
        'description': '对输入张量进行 d 天的延迟处理',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            }
        }
    },
    'D_ts_delta': {
        'description': '计算输入张量与 d 天前的差值',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            }
        }
    },
    'D_ts_pctchg': {
        'description': '计算输入张量在回溯 lookback 天内的涨跌幅',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            }
        }
    },
    'D_ts_mean': {
        'description': '计算输入张量在回溯 lookback 天内的均值',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            }
        }
    },
    'D_ts_harmonic_mean': {
        'description': '计算输入张量在回溯 lookback 天内的调和平均值',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            }
        }
    },
    'D_ts_std': {
        'description': '计算输入张量在回溯 lookback 天内的标准差',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            }
        }
    },
    'D_ts_to_max': {
        'description': '计算 x 与回溯 lookback 天内的最大值的比值',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            }
        }
    },
    'D_ts_to_min': {
        'description': '计算 x 与回溯 lookback 天内的最小值的比值',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            }
        }
    },
    'D_ts_to_mean': {
        'description': '计算 x 与回溯 lookback 天内的均值的比值',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            }
        }
    },
    'D_ts_max_to_min': {
        'description': '计算 x 与回溯 lookback 天内的最大值与最小值的差值的比值',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            }
        }
    },
    'D_ts_maxmin_norm': {
        'description': '对 x 进行回溯 lookback 天内的最大最小值归一化',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            }
        }
    },
    'D_ts_norm': {
        'description': '对 x 进行回溯 lookback 天内的 z-score 标准化',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            }
        }
    },
    'D_ts_detrend': {
        'description': '去除 x 在回溯 lookback 天内的趋势',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            }
        }
    },
    'D_ts_mask_mean': {
        'description': '对 mask 为 True 的部分计算 x 的均值',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AC属'
            }
        }
    },
    'D_ts_mask_std': {
        'description': '对 mask 为 True 的部分计算 x 的标准差',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AC属'
            }
        }
    },
    'D_ts_mask_sum': {
        'description': '对 mask 为 True 的部分计算 x 的和',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AC属'
            }
        }
    },
    'D_ts_mask_prod': {
        'description': '对 mask 为 True 的部分计算 x 的乘积',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AC属'
            }
        }
    },
    'D_Minute_area_mean': {
        'description': '对 mask 为 True 的部分计算 x 的均值',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',  # 修正为混频科
                '属': 'BD属'
            }
        }
    },
    'D_Minute_area_std': {
        'description': '对 mask 为 True 的部分计算 x 的标准差',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',  # 修正为混频科
                '属': 'BD属'
            }
        }
    },
    'D_Minute_area_sum': {
        'description': '对 mask 为 True 的部分计算 x 的和',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',  # 修正为混频科
                '属': 'BD属'
            }
        }
    },
    'D_Minute_area_prod': {
        'description': '对 mask 为 True 的部分计算 x 的乘积',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',  # 修正为混频科
                '属': 'BD属'
            }
        }
    },
    'D_Minute_std': {
        'description': '计算日内数据的标准差',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',
                '属': 'B属'
            }
        }
    },
    'D_Minute_mean': {
        'description': '计算日内数据的均值',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',
                '属': 'B属'
            }
        }
    },
    'D_Minute_trend': {
        'description': '计算日内数据的变化趋势',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',
                '属': 'B属'
            }
        }
    },
    'D_Minute_area_weight_mean': {
        'description': '对 mask 为 True 的部分，以 weight 为权重计算 x 的加权均值',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',  # 修正为混频科
                '属': 'BBD属'
            }
        }
    },
    'D_Minute_area_corr': {
        'description': '对 mask 为 True 的部分计算 x 和 y 的相关性',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',  # 修正为混频科
                '属': 'BBD属'
            }
        }
    },
    'D_Minute_area_rankcorr': {
        'description': '对 mask 为 True 的部分计算 x 和 y 的秩相关性',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',  # 修正为混频科
                '属': 'BBD属'
            }
        }
    },
    'D_Minute_area_bifurcate_mean': {
        'description': '根据 m_tensor_y 的均值将 m_tensor_x 分为两部分，分别计算均值并相减',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',  # 修正为混频科
                '属': 'BBD属'
            }
        }
    },
    'D_Minute_area_bifurcate_std': {
        'description': '根据 m_tensor_y 的均值将 m_tensor_x 分为两部分，分别计算标准差并相减',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',  # 修正为混频科
                '属': 'BBD属'
            }
        }
    },
    'D_Minute_corr': {
        'description': '计算 x 和 y 的相关性',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',
                '属': 'BB属'
            }
        }
    },
    'D_Minute_weight_mean': {
        'description': '以 weight 为权重计算 x 的加权均值',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',
                '属': 'BB属'
            }
        }
    },
    'D_Minute_abnormal_point_count': {
        'description': '计算 mask 中异常点的数量',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',
                '属': 'D属'
            }
        }
    },


    'M_ignore_wobble': {
        'description': '将开盘前 window_size 分钟和收盘前 window_size 分钟的数据设置为 NaN',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'B属'
            }
        }
    },
    'M_cs_zscore': {
        'description': '对每个股票的每日数据进行 z-score 标准化',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'B属'
            }
        }
    },
    'M_cs_rank': {
        'description': '对每个股票的每日数据进行排名标准化，范围为 [0, 1]',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'B属'
            }
        }
    },
    'M_cs_scale': {
        'description': '对每个股票的每日数据进行最大值最小值标准化，范围为 [0, 1]',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'B属'
            }
        }
    },
    'M_cs_demean': {
        'description': '对每个股票的每日数据减去其均值',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'B属'
            }
        }
    },
    'M_cs_winsor': {
        'description': '对每个股票的每日数据进行 Winsor 处理，将超出百分位数的值替换为对应百分位数的值',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'B属'
            }
        }
    },
    'M_at_abs': {
        'description': '计算输入张量的绝对值',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'B属'
            }
        }
    },
    'M_ts_delay': {
        'description': '对输入张量进行 d 天的延迟处理',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'B属'
            }
        }
    },
    'M_ts_pctchg': {
        'description': '计算输入张量在回溯 lookback 天内的涨跌幅',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'B属'
            }
        }
    },
    'M_at_add': {
        'description': '对 x 和 y 进行逐元素相加',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BB属'
            }
        }
    },
    'M_at_sub': {
        'description': '对 x 和 y 进行逐元素相减',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BB属'
            }
        }
    },
    'M_at_div': {
        'description': '对 x 和 y 进行逐元素相除，y 为 0 时返回 NaN',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BB属'
            }
        }
    },
    'M_at_sign': {
        'description': '计算输入张量的符号函数，返回 -1、0 或 1',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BB属'
            }
        }
    },
    'M_cs_cut': {
        'description': '根据 x 的符号对 y 进行截断处理',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BB属'
            }
        }
    },
    'M_cs_umr': {
        'description': '计算 (x - mean(y)) * y',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BB属'
            }
        }
    },
    'M_at_prod': {
        'description': '对 x 和 y 进行逐元素相除，y 为 0 或 NaN 时返回 NaN',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BB属'
            }
        }
    },
    'M_cs_norm_spread': {
        'description': '计算 (x - y) / (|x| + |y|)',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BB属'
            }
        }
    },
    'M_toD_standard': {
        'description': '将 M_tensor 除以 D_tensor 的转置并扩展维度后的结果',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '混频科',  # 修正为混频科
                '属': 'BA属'
            }
        }
    },
    'M_cs_edge_flip': {
        'description': '根据阈值 thresh 对 M_tensor 的边缘部分进行翻转处理',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BG属'
            }
        }
    },
    'M_ts_delta': {
        'description': '计算 m_tensor 与回溯 lookback 天前的差值',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BF属'
            }
        }
    },
    'M_ts_mean_left_neighbor': {
        'description': '计算 m_tensor 左侧邻居的均值',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BF属'
            }
        }
    },
    'M_ts_mean_mid_neighbor': {
        'description': '计算 m_tensor 中间邻居的均值',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BF属'
            }
        }
    },
    'M_ts_mean_right_neighbor': {
        'description': '计算 m_tensor 右侧邻居的均值',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BF属'
            }
        }
    },
    'M_ts_std_left_neighbor': {
        'description': '计算 m_tensor 左侧邻居的标准差',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BF属'
            }
        }
    },
    'M_ts_std_mid_neighbor': {
        'description': '计算 m_tensor 中间邻居的标准差',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BF属'
            }
        }
    },
    'M_ts_std_right_neighbor': {
        'description': '计算 m_tensor 右侧邻居的标准差',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BF属'
            }
        }
    },
    'M_ts_product_left_neighbor': {
        'description': '计算 m_tensor 左侧邻居的乘积',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BF属'
            }
        }
    },
    'M_ts_product_mid_neighbor': {
        'description': '计算 m_tensor 中间邻居的乘积',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BF属'
            }
        }
    },
    'M_ts_product_right_neighbor': {
        'description': '计算 m_tensor 右侧邻居的乘积',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BF属'
            }
        }
    },

    'Dmask_min': {
        'description': '对 x 进行 unfold 操作，回溯 lookback 天的数据，取最小的 1/4 天作为掩码',
        'classification': {
            'interface': {
                '目': 'C目',
                '科': '同频科',
                '属': 'AF属'
            }
        }
    },
    'Dmask_max': {
        'description': '对 x 进行 unfold 操作，回溯 lookback 天的数据，取最大的 1/4 天作为掩码',
        'classification': {
            'interface': {
                '目': 'C目',
                '科': '同频科',
                '属': 'AF属'
            }
        }
    },
    'Dmask_middle': {
        'description': '对 x 进行 unfold 操作，回溯 lookback 天的数据，取中间的 1/2 天作为掩码',
        'classification': {
            'interface': {
                '目': 'C目',
                '科': '同频科',
                '属': 'AF属'
            }
        }
    },
    'Dmask_mean_plus_std': {
        'description': '对 x 进行 unfold 操作，回溯 lookback 天的数据，进行标准化处理，取大于均值 + 标准差的部分作为掩码',
        'classification': {
            'interface': {
                '目': 'C目',
                '科': '同频科',
                '属': 'AF属'
            }
        }
    },
    'Dmask_mean_sub_std': {
        'description': '对 x 进行 unfold 操作，回溯 lookback 天的数据，进行标准化处理，取小于均值 - 标准差的部分作为掩码',
        'classification': {
            'interface': {
                '目': 'C目',
                '科': '同频科',
                '属': 'AF属'
            }
        }
    },


    "Mmask_min": {
        "description": "返回日内数据的最小 1/4 部分作为掩码",
        "classification": {
            "interface": {
                "目": "D目",
                "科": "同频科",
                "属": "B属"
            }
        }
    },
    "Mmask_max": {
        "description": "返回日内数据的最大 1/4 部分作为掩码",
        "classification": {
            "interface": {
                "目": "D目",
                "科": "同频科",
                "属": "B属"
            }
        }
    },
    "Mmask_middle": {
        "description": "返回日内数据的中间 1/2 部分作为掩码",
        "classification": {
            "interface": {
                "目": "D目",
                "科": "同频科",
                "属": "B属"
            }
        }
    },
    "Mmask_min_to_max": {
        "description": "返回日内数据的最小值和最大值之间的部分作为掩码",
        "classification": {
            "interface": {
                "目": "D目",
                "科": "同频科",
                "属": "B属"
            }
        }
    },
    "Mmask_mean_plus_std": {
        "description": "返回日内数据标准化后大于均值 + 标准差的部分作为掩码",
        "classification": {
            "interface": {
                "目": "D目",
                "科": "同频科",
                "属": "B属"
            }
        }
    },
    "Mmask_mean_sub_std": {
        "description": "返回日内数据标准化后小于均值 - 标准差的部分作为掩码",
        "classification": {
            "interface": {
                "目": "D目",
                "科": "同频科",
                "属": "B属"
            }
        }
    },
    "Mmask_1h_after_open": {
        "description": "返回开盘后第 1 个小时的数据",
        "classification": {
            "interface": {
                "目": "D目",
                "科": "同频科",
                "属": "B属"
            }
        }
    },
    "Mmask_1h_before_close": {
        "description": "返回收盘前第 1 个小时的数据",
        "classification": {
            "interface": {
                "目": "D目",
                "科": "同频科",
                "属": "B属"
            }
        }
    },
    "Mmask_2h_middle": {
        "description": "返回中间 2 个小时的数据",
        "classification": {
            "interface": {
                "目": "D目",
                "科": "同频科",
                "属": "B属"
            }
        }
    },
    "Mmask_morning": {
        "description": "返回早上 2 个小时的数据",
        "classification": {
            "interface": {
                "目": "D目",
                "科": "同频科",
                "属": "B属"
            }
        }
    },
    "Mmask_afternoon": {
        "description": "返回下午 2 个小时的数据",
        "classification": {
            "interface": {
                "目": "D目",
                "科": "同频科",
                "属": "B属"
            }
        }
    },
    "Mmask_day_plus": {
        "description": "返回大于日频数据的部分作为掩码",
        "classification": {
            "interface": {
                "目": "D目",
                "科": "混频科",  # 修正为混频科
                "属": "BA属"
            }
        }
    },
    "Mmask_day_sub": {
        "description": "返回小于日频数据的部分作为掩码",
        "classification": {
            "interface": {
                "目": "D目",
                "科": "混频科",  # 修正为混频科
                "属": "BA属"
            }
        }
    },
    "Mmask_rolling_plus": {
        "description": "以日内数据最大 1/4 部分的均值作为日较大值，返回大于 lookback 期内最大日较大值的部分作为掩码",
        "classification": {
            "interface": {
                "目": "D目",
                "科": "同频科",  
                "属": "BF属"
            }
        }
    },
    "Mmask_rolling_sub": {
        "description": "以日内数据最小 1/4 部分的均值作为日较大值，返回小于 lookback 期内最小日较大值的部分作为掩码",
        "classification": {
            "interface": {
                "目": "D目",
                "科": "同频科",  
                "属": "BF属"
            }
        }
    },
    "Mmask_and": {
        "description": "对两个掩码张量进行逻辑与运算",
        "classification": {
            "interface": {
                "目": "D目",
                "科": "同频科",
                "属": "DD属"
            }
        }
    },
    "Mmask_or": {
        "description": "对两个掩码张量进行逻辑或运算",
        "classification": {
            "interface": {
                "目": "D目",
                "科": "同频科",
                "属": "DD属"
            }
        }
    }
    
}
