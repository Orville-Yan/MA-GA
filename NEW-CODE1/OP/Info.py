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

VariableType_info = {
    'TypeA': {'shape': '(day_len, num_stock)的二维tensor',
              'name': 'day_OHLCV',
              '数据': '日频的高开低收和成交量数据，即D_O,D_H,D_L,D_C,D_V'
              },
    'TypeB': {'shape': '(num_stock, day_len, minute_len=240)的三维tensor',
              'name': 'minute_OHLCV',
              '数据': '分钟频的高开低收和成交量数据，即M_O,M_H,M_L,M_C,M_V'
              },
    'TypeC': {'shape': '(day_len, num_stock, rolling_day)的三维tensor',
              'name': 'day_mask',
              '数据': 'TypeA的掩码'
              },
    'TypeD': {'shape': '(num_stock, day_len, minute_len=240)的三维tensor',
              'name': 'minute_mask',
              '数据': 'TypeB的掩码'
              },
    'TypeE': {'shape': '(day_len, num_stock, industry_num=31)的三维tensor',
              'name': 'industry',
              '数据': '01的哑变量矩阵，表示是否属于该行业'
              },
    'TypeF': {'shape': '一维tensor',
              'name': 'time_int',
              '数据': '1，2，3，5，10，20，30，60'
              },
    'TypeG': {'shape': 'float变量',
              'name': 'threshold',
              '数据': '0.05，0.1'
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
        'description': '输入一个A类型的张量，计算并返回其绝对值。此操作用于消除负号的影响，保留数值的大小。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'A属'
            },
            'action': {
                '扭曲'
            }
        }
    },
    'D_cs_demean_industry': {
        'description': '输入一个A类型的张量和一个E类型的行业哑变量矩阵。计算每个行业的均值，并对每个股票的日频数据进行去均值处理，返回绝对值。此操作有助于消除行业间的系统性差异。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AE属'
            },
            'action': {
                '剥离'
            }
        }
    },
    'D_cs_industry_neutra': {
        'description': '输入一个A类型的张量和一个E类型的行业哑变量矩阵。对每个股票的日频数据进行行业中性化处理，去除行业均值的影响，使得数据更具可比性。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AE属'
            },
            'action': {
                '剥离'
            }
        }
    },
    'D_cs_norm_spread': {
        'description': '输入两个A类型的张量x和y。计算 (x - y) / (|x| + |y|)，用于衡量两个变量之间的相对差异，避免因绝对值大小不同而导致的偏差。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AA属'
            },
            'action': {
                '混沌'
            }
        }
    },
    'D_cs_cut': {
        'description': '输入两个A类型的张量x和y。根据x的符号对y进行截断处理，x为正则保留y的值，为负则将y置为0。此操作用于根据x的变化方向调整y的值。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AA属'
            },
            'action': {
                '混沌'
            }
        }
    },
    'D_cs_regress_res': {
        'description': '输入两个A类型的张量x和y。对x和y进行截面回归，返回残差。此操作用于去除线性趋势，保留数据的非线性部分。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AA属'
            },
            'action': {
                '剥离'
            }
        }
    },
    'D_at_add': {
        'description': '输入两个A类型的张量x和y。逐元素相加，返回结果。此操作用于合并两个张量的数值。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AA属'
            },
            'action': {
                '混沌'
            }
        }
    },
    'D_at_sub': {
        'description': '输入两个A类型的张量x和y。逐元素相减，返回结果。此操作用于计算两个张量之间的差异。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AA属'
            },
            'action': {
                '混沌'
            }
        }
    },
    'D_at_div': {
        'description': '输入两个A类型的张量x和y。逐元素相除，y为0时返回NaN。此操作用于计算两个张量的比值。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AA属'
            },
            'action': {
                '混沌'
            }
        }
    },
    'D_at_prod': {
        'description': '输入两个A类型的张量x和y。逐元素相乘，返回结果。此操作用于计算两个张量的乘积。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AA属'
            },
            'action': {
                '混沌'
            }
        }
    },
    'D_at_mean': {
        'description': '输入两个A类型的张量x和y。计算逐元素均值，返回结果。此操作用于平滑两个张量的数值。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AA属'
            },
            'action': {
                '描述'
            }
        }
    },
    'D_cs_edge_flip': {
        'description': '输入一个A类型的张量x和一个G类型的阈值。对x的分位数处于阈值以外的部分取负号。此操作用于突出数据的极端值。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AG属'
            },
            'action': {
                '扭曲'
            }
        }
    },
    'D_ts_corr': {
        'description': '输入两个A类型的张量x和y，以及一个F类型的int参数d。计算x和y在回溯d天内的时序相关性，用于评估两个时间序列之间的线性关系。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AAF属'
            },
            'action': {
                '描述'
            }
        }
    },
    'D_ts_rankcorr': {
        'description': '输入两个A类型的张量x和y，以及一个F类型的int参数d。计算x和y在回溯d天内的时序秩相关性，适用于非线性关系的评估。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AAF属'
            },
            'action': {
                '描述'
            }
        }
    },
    'D_ts_regress_res': {
        'description': '输入两个A类型的张量x和y，以及一个F类型的int参数d。对x和y进行时序回归，返回残差。此操作用于去除时间序列中的线性趋势。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AAF属'
            },
            'action': {
                '描述', '剥离'
            }
        }
    },
    'D_ts_weight_mean': {
        'description': '输入两个A类型的张量x和y，以及一个F类型的int参数d。以y为权重，计算x在回溯d天内的加权平均值，适用于强调某些数据点的重要性。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AAF属'
            },
            'action': {
                '描述','自适应'
            }
        }
    },
    'D_ts_max': {
        'description': '输入一个A类型的张量x和一个F类型的int参数d。计算x在回溯d天内的最大值，用于识别时间序列中的峰值。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            },
            'action': {
                '描述','自适应'
            }
        }
    },
    'D_ts_min': {
        'description': '输入一个A类型的张量x和一个F类型的int参数d。计算x在回溯d天内的最小值，用于识别时间序列中的谷值。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            },
            'action': {
                '描述','自适应'
            }
        }
    },
    'D_ts_delay': {
        'description': '输入一个A类型的张量x和一个F类型的int参数d。对输入张量进行d天的延迟处理，常用于构建滞后变量。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            },
            'action': {
                '混沌'
            }
        }
    },
    'D_ts_delta': {
        'description': '输入一个A类型的张量x和一个F类型的int参数d。计算输入张量与d天前的差值，用于衡量时间序列的变化幅度。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            },
            'action': {
                '混沌'
            }
        }
    },
    'D_ts_pctchg': {
        'description': '输入一个A类型的张量x和一个F类型的int参数d。计算输入张量在回溯d天内的涨跌幅，适用于分析时间序列的相对变化。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            },
            'action': {
                '描述'
            }
        }
    },
    'D_ts_mean': {
        'description': '输入一个A类型的张量x和一个F类型的int参数d。计算输入张量在回溯d天内的均值，用于平滑时间序列。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            },
            'action': {
                '描述'
            }
        }
    },
    'D_ts_harmonic_mean': {
        'description': '输入一个A类型的张量x和一个F类型的int参数d。计算输入张量在回溯d天内的调和平均值，用于平滑时间序列中的极端值。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            },
            'action': {
                '描述'
            }
        }
    },
    'D_ts_std': {
        'description': '输入一个A类型的张量x和一个F类型的int参数d。计算输入张量在回溯d天内的标准差，用于评估时间序列的波动性。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            },
            'action': {
                '描述'
            }
        }
    },
    'D_ts_to_max': {
        'description': '输入一个A类型的张量x和一个F类型的int参数d。计算x与x回溯d天内的最大值的比值。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            },
            'action': {
                '自适应'
            }
        }
    },
    'D_ts_to_min': {
        'description': '输入一个A类型的张量x和一个F类型的int参数d。计算x与x回溯d天内的最小值的比值。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            },
            'action': {
                '自适应'
            }
        }
    },
    'D_ts_to_mean': {
        'description': '输入一个A类型的张量x和一个F类型的int参数d。计算x与x回溯d天内的均值的比值。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            },
            'action': {
                '自适应'
            }
        }
    },
    'D_ts_max_to_min': {
        'description': '输入一个A类型的张量x和一个F类型的int参数d。计算x与x回溯d天内的最大值与最小值的差值的比值。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            },
            'action': {
                '自适应'
            }
        }
    },
    'D_ts_maxmin_norm': {
        'description': '输入一个A类型的张量x和一个F类型的int参数d。对x回溯d天，做最大最小值归一化。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            },
            'action': {
                '自适应'
            }
        }
    },
    'D_ts_norm': {
        'description': '输入一个A类型的张量x和一个F类型的int参数d。对x进行回溯d天内的z-score标准化。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            },
            'action': {
                '自适应'
            }
        }
    },
    'D_ts_detrend': {
        'description': '输入一个A类型的张量x和一个F类型的int参数d。去除x在回溯d天内的趋势，返回残差。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            },
            'action': {
                '剥离'
            }
        }
    },
    'D_ts_mask_mean': {
        'description': '输入A类型的张量x和C类型的mask。对mask为True的部分计算x的均值。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AC属'
            },
            'action': {
                '提纯'
            }
        }
    },
    'D_ts_mask_std': {
        'description': '输入A类型的张量x和C类型的mask。对mask为True的部分计算x的标准差。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AC属'
            },
            'action': {
                '提纯'
            }
        }
    },
    'D_ts_mask_sum': {
        'description': '输入A类型的张量x和C类型的mask。对mask为True的部分计算x的和。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AC属'
            },
            'action': {
                '提纯'
            }
        }
    },
    'D_ts_mask_prod': {
        'description': '输入A类型的张量x和C类型的mask。对mask为True的部分计算x的乘积。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AC属'
            },
            'action': {
                '提纯'
            }
        }
    },
    'D_Minute_area_mean': {
        'description': '输入B类型的张量x和D类型的mask。对mask为True的部分计算x的均值。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',
                '属': 'BD属'
            },
            'action': {
                '提纯'
            }
        }
    },
    'D_Minute_area_std': {
        'description': '输入B类型的张量x和D类型的mask。对mask为True的部分计算x的标准差。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',
                '属': 'BD属'
            },
            'action': {
                '提纯'
            }
        }
    },
    'D_Minute_area_sum': {
        'description': '输入B类型的张量x和D类型的mask。对mask为True的部分计算x的和。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',
                '属': 'BD属'
            },
            'action': {
                '提纯'
            }
        }
    },
    'D_Minute_area_prod': {
        'description': '输入B类型的张量x和D类型的mask。对mask为True的部分计算x的乘积。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',
                '属': 'BD属'
            },
            'action': {
                '提纯'
            }
        }
    },
    'D_Minute_std': {
        'description': '输入B类型的张量。计算日内分钟数据的标准差。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',
                '属': 'B属'
            },
            'action': {
                '描述'
            }
        }
    },
    'D_Minute_mean': {
        'description': '输入B类型的张量。计算日内分钟数据的均值。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',
                '属': 'B属'
            },
            'action': {
                '描述'
            }
        }
    },
    'D_Minute_trend': {
        'description': '输入B类型的张量。计算日内数据的变化趋势。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',
                '属': 'B属'
            },
            'action': {
                '描述'
            }
        }
    },
    'D_Minute_area_weight_mean': {
        'description': '对 日内数据 mask 为 True 的部分，以 weight 为权重计算 x 的加权均值，用于在掩码条件下的加权分析。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',
                '属': 'BBD属'
            },
            'action': {
                '描述','提纯'
            }
        }
    },
    'D_Minute_area_corr': {
        'description': '对 日内数据 mask 为 True 的部分计算 x 和 y 的相关性，用于在掩码条件下的相关性分析。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',
                '属': 'BBD属'
            },
            'action': {
                '描述','提纯'
            }
        }
    },
    'D_Minute_area_rankcorr': {
        'description': '对 日内数据 mask 为 True 的部分计算 x 和 y 的秩相关性，用于在掩码条件下的秩相关性分析。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',
                '属': 'BBD属'
            },
            'action': {
                '描述','提纯'
            }
        }
    },
    'D_Minute_area_bifurcate_mean': {
        'description': '根据 m_tensor_y 的均值将 m_tensor_x 分为两部分，分别计算均值并相减，用于分析日内数据的分叉特性。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',
                '属': 'BBD属'
            },
            'action': {
                '描述','提纯'
            }
        }
    },
    'D_Minute_area_bifurcate_std': {
        'description': '根据 m_tensor_y 的均值将 m_tensor_x 分为两部分，分别计算标准差并相减，用于分析日内数据的分叉波动性。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',
                '属': 'BBD属'
            },
            'action': {
                '描述','提纯'
            }
        }
    },
    'D_Minute_corr': {
        'description': '计算日内数据 x 和 y 的相关性，用于评估日内数据的线性关系。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'BB属'
            },
            'action': {
                '描述'
            }
        }
    },
    'D_Minute_weight_mean': {
        'description': '以 weight 为权重计算日内数据 x 的加权均值，用于强调某些时间点的重要性。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'BB属'
            },
            'action': {
                '描述'
            }
        }
    },
    'D_Minute_abnormal_point_count': {
        'description': '输入D类型的mask。计算mask中标记为True的数量。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',
                '属': 'D属'
            },
            'action': {
                '提纯'
            }
        }
    },
    'M_ignore_wobble': {
        'description': '输入B类型的张量。将开盘后5分钟和收盘前5分钟的数据设置为NaN。',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'B属'
            },
            'action': {
                '提纯'
            }
        }
    },
    'M_cs_zscore': {
        'description': '输入B类型的张量。对每个股票的每日数据进行z-score标准化。',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'B属'
            },
            'action': {
                '描述'
            }
        }
    },
    'M_cs_rank': {
        'description': '输入B类型的张量。计算每一分钟数据在日内的分位数。',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'B属'
            },
            'action': {
                '描述'
            }
        }
    },
    'M_cs_scale': {
        'description': '输入B类型的张量。对每个股票的每日数据进行最大值最小值标准化。',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'B属'
            },
            'action': {
                '描述'
            }
        }
    },
    'M_cs_demean': {
        'description': '输入B类型的张量。对每个股票的每日数据减去其均值。',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'B属'
            },
            'action': {
                '剥离'
            }
        }
    },
    'M_cs_winsor': {
        'description': '输入B类型的张量。对每个股票的每日数据进行Winsor处理。',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'B属'
            },
            'action': {
                '剥离'
            }
        }
    },
    'M_at_abs': {
        'description': '输入B类型的张量。计算输入张量的绝对值。',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'B属'
            },
            'action': {
                '扭曲'
            }
        }
    },
    'M_ts_delay': {
        'description': '输入B类型的张量。获得1分钟之前的数据。',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BF属'
            },
            'action': {
                '混沌'
            }
        }
    },
    'M_ts_pctchg': {
        'description': '计算输入张量在回溯 lookback 天内的涨跌幅',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BF属'
            },
            'action': {
                '描述'
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
            },
            'action': {
                '混沌'
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
            },
            'action': {
                '混沌'
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
            },
            'action': {
                '混沌'
            }
        }
    },
    'M_at_sign': {
        'description': '计算输入张量的符号函数，返回 -1、0 或 1',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'B属'
            },
            'action': {
                '共轭'
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
            },
            'action': {
                '混沌'
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
            },
            'action': {
                '混沌'
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
            },
            'action': {
                '混沌'
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
            },
            'action': {
                '混沌'
            }
        }
    },
    'M_toD_standard': {
        'description': '将 M_tensor 除以 D_tensor 的转置并扩展维度后的结果',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '混频科',
                '属': 'BA属'
            },
            'action': {
                '扭曲'
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
            },
            'action': {
                '扭曲'
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
            },
            'action': {
                '混沌'
            }
        }
    },
    'M_ts_mean_left_neighbor': {
        'description': '计算输入张量在时间维度上向左移动 neighbor_range 步后的均值',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BF属'
            },
            'action': {
                '描述'
            }
        }
    },
    'M_ts_mean_mid_neighbor': {
        'description': '计算输入张量在时间维度上每个时间点的邻域均值，保留中间部分，两端填充 NaN',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BF属'
            },
            'action': {
                '描述'
            }
        }
    },
    'M_ts_mean_right_neighbor': {
        'description': '计算输入张量在时间维度上向右移动 neighbor_range 步后的均值',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BF属'
            },
            'action': {
                '描述'
            }
        }
    },
    'M_ts_std_left_neighbor': {
        'description': '计算输入张量在时间维度上向左移动 neighbor_range 步后的标准差',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BF属'
            },
            'action': {
                '描述'
            }
        }
    },
    'M_ts_std_mid_neighbor': {
        'description': '计算输入张量在时间维度上每个时间点的邻域标准差，保留中间部分，两端填充 NaN',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BF属'
            },
            'action': {
                '描述'
            }
        }
    },
    'M_ts_std_right_neighbor': {
        'description': '计算输入张量在时间维度上向右移动 neighbor_range 步后的标准差',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BF属'
            },
            'action': {
                '描述'
            }
        }
    },
    'M_ts_product_left_neighbor': {
        'description': '计算输入张量在时间维度上向左移动 neighbor_range 步后的乘积',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BF属'
            },
            'action': {
                '混沌'
            }
        }
    },
    'M_ts_product_mid_neighbor': {
        'description': '计算输入张量在时间维度上每个时间点的邻域乘积，保留中间部分，两端填充 NaN积',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BF属'
            },
            'action': {
                '混沌'
            }
        }
    },
    'M_ts_product_right_neighbor': {
        'description': '计算输入张量在时间维度上向右移动 neighbor_range 步后的乘积',
        'classification': {
            'interface': {
                '目': 'B目',
                '科': '同频科',
                '属': 'BF属'
            },
            'action': {
                '混沌'
            }
        }
    },
    'Dmask_min': {
        'description': '输入一个A类型的张量x和一个F类型的int参数d。将x展开成一个三维的张量，第三维的长度为d。将第三维中数值最小的1/4标记为True，其余为False。',
        'classification': {
            'interface': {
                '目': 'C目',
                '科': '同频科',
                '属': 'AF属'
            },
            'action': {
                '提纯'
            }
        }
    },
    'Dmask_max': {
        'description': '输入一个A类型的张量x和一个F类型的int参数d。将x展开成一个三维的张量，第三维的长度为d。将第三维中数值最大的1/4标记为True，其余为False。',
        'classification': {
            'interface': {
                '目': 'C目',
                '科': '同频科',
                '属': 'AF属'
            },
            'action': {
                '提纯'
            }
        }
    },
    'Dmask_middle': {
        'description': '输入一个A类型的张量x和一个F类型的int参数d。将x展开成一个三维的张量，第三维的长度为d。将第三维中数值位于中间1/2的部分标记为True，其余为False。',
        'classification': {
            'interface': {
                '目': 'C目',
                '科': '同频科',
                '属': 'AF属'
            },
            'action': {
                '提纯'
            }
        }
    },
    'Dmask_mean_plus_std': {
        'description': '输入一个A类型的张量x和一个F类型的int参数d。将x展开成一个三维的张量，第三维的长度为d。将第三维中数值大于均值加上一个标准差的部分标记为True，其余为False。',
        'classification': {
            'interface': {
                '目': 'C目',
                '科': '同频科',
                '属': 'AF属'
            },
            'action': {
                '描述', '提纯'
            }
        }
    },
    'Dmask_mean_sub_std': {
        'description': '输入一个A类型的张量x和一个F类型的int参数d。将x展开成一个三维的张量，第三维的长度为d。将第三维中数值小于均值减去一个标准差的部分标记为True，其余为False。',
        'classification': {
            'interface': {
                '目': 'C目',
                '科': '同频科',
                '属': 'AF属'
            },
            'action': {
                '描述', '提纯'
            }
        }
    },
    'Mmask_min': {
        'description': '输入一个B类型的张量。将日内数值最小的1/4标记为True，其余为False。',
        'classification': {
            'interface': {
                '目': 'D目',
                '科': '同频科',
                '属': 'B属'
            },
            'action': {
                '提纯'
            }
        }
    },
    'Mmask_max': {
        'description': '输入一个B类型的张量。将日内数值最大的1/4标记为True，其余为False。',
        'classification': {
            'interface': {
                '目': 'D目',
                '科': '同频科',
                '属': 'B属'
            },
            'action': {
                '提纯'
            }
        }
    },
    'Mmask_middle': {
        'description': '输入一个B类型的张量。将日内数值大小位于中间1/2的部分标记为True，其余为False。',
        'classification': {
            'interface': {
                '目': 'D目',
                '科': '同频科',
                '属': 'B属'
            },
            'action': {
                '提纯'
            }
        }
    },
    'Mmask_min_to_max': {
        'description': '输入一个B类型的张量。标记日内最大值和最小值之间的所有分钟为True，其余为False。',
        'classification': {
            'interface': {
                '目': 'D目',
                '科': '同频科',
                '属': 'B属'
            },
            'action': {
                '提纯'
            }
        }
    },
    'Mmask_mean_plus_std': {
        'description': '输入一个B类型的张量。将日内数值大于均值加上一个标准差的部分标记为True，其余为False。',
        'classification': {
            'interface': {
                '目': 'D目',
                '科': '同频科',
                '属': 'B属'
            },
            'action': {
                '描述', '提纯'
            }
        }
    },
    'Mmask_mean_sub_std': {
        'description': '输入一个B类型的张量。将日内数值小于均值减去一个标准差的部分标记为True，其余为False。',
        'classification': {
            'interface': {
                '目': 'D目',
                '科': '同频科',
                '属': 'B属'
            },
            'action': {
                '描述', '提纯'
            }
        }
    },
    'Mmask_1h_after_open': {
        'description': '输入一个B类型的张量。将每一天开盘后的第一个小时标记为True，其余为False。',
        'classification': {
            'interface': {
                '目': 'D目',
                '科': '同频科',
                '属': 'B属'
            },
            'action': {
                '提纯'
            }
        }
    },
    'Mmask_1h_before_close': {
        'description': '输入一个B类型的张量。将每一天收盘前的最后一个小时标记为True，其余为False。',
        'classification': {
            'interface': {
                '目': 'D目',
                '科': '同频科',
                '属': 'B属'
            },
            'action': {
                '提纯'
            }
        }
    },
    'Mmask_2h_middle': {
        'description': '输入一个B类型的张量。将每一天的中间两个小时标记为True，其余为False。',
        'classification': {
            'interface': {
                '目': 'D目',
                '科': '同频科',
                '属': 'B属'
            },
            'action': {
                '提纯'
            }
        }
    },
    'Mmask_morning': {
        'description': '输入一个B类型的张量。将每一天的早上两个小时标记为True，其余为False。',
        'classification': {
            'interface': {
                '目': 'D目',
                '科': '同频科',
                '属': 'B属'
            },
            'action': {
                '提纯'
            }
        }
    },
    'Mmask_afternoon': {
        'description': '输入一个B类型的张量。将每一天的下午两个小时标记为True，其余为False。',
        'classification': {
            'interface': {
                '目': 'D目',
                '科': '同频科',
                '属': 'B属'
            },
            'action': {
                '提纯'
            }
        }
    },
    'Mmask_day_plus': {
        'description': '输入一个B类型的张量x和一个A类型的张量y。x某天的分钟数据，以y那一天的数据为基准，若大于则标记为True，其余为False。',
        'classification': {
            'interface': {
                '目': 'D目',
                '科': '混频科',
                '属': 'BA属'
            },
            'action': {
                '提纯'
            }
        }
    },
    'Mmask_day_sub': {
        'description': '输入一个B类型的张量x和一个A类型的张量y。x某天的分钟数据，以y那一天的数据为基准，若小于则标记为True，其余为False。',
        'classification': {
            'interface': {
                '目': 'D目',
                '科': '混频科',
                '属': 'BA属'
            },
            'action': {
                '提纯'
            }
        }
    },
    'Mmask_rolling_plus': {
        'description': '输入一个B类型的张量x和一个F类型的int参数d。以日内数据最大1/4部分的均值作为日较大值，回溯d天，取d个较大值中的最大值，将大于最大值的部分标记为True，其余为False。',
        'classification': {
            'interface': {
                '目': 'D目',
                '科': '同频科',
                '属': 'BF属'
            },
            'action': {
                '描述', '提纯'
            }
        }
    },
    'Mmask_rolling_sub': {
        'description': '输入一个B类型的张量x和一个F类型的int参数d。以日内数据最小1/4部分的均值作为日较小值，回溯d天，取d个较小值中的最小值，将小于最小值的部分标记为True，其余为False。',
        'classification': {
            'interface': {
                '目': 'D目',
                '科': '同频科',
                '属': 'BF属'
            },
            'action': {
                '描述', '提纯'
            }
        }
    },
    'Mmask_and': {
        'description': '输入两个D类型的mask。对两个mask进行逐元素的逻辑与运算，只有当两个mask均为True时，结果为True。',
        'classification': {
            'interface': {
                '目': 'D目',
                '科': '同频科',
                '属': 'DD属'
            },
            'action': {
                '提纯'
            }
        }
    },
    'Mmask_or': {
        'description': '输入两个D类型的mask。对两个mask进行逐元素的逻辑或运算，只要有一个mask为True，结果为True。',
        'classification': {
            'interface': {
                '目': 'D目',
                '科': '同频科',
                '属': 'DD属'
            },
            'action': {
                '提纯'
            }
        }
    }
}
