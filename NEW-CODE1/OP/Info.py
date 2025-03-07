data_info = {
    'D_O': {
        'shape': '(day_len,num_stock)',
        'description': '当日开盘价'
    },
    'D_C': {
        'shape': '(day_len,num_stock)',
        'description': '当日收盘价'
    },
    'D_H': {
        'shape': '(day_len,num_stock)',
        'description': '当日最高价'
    },
    'D_L': {
        'shape': '(day_len,num_stock)',
        'description': '当日最低价'
    },
    'D_V': {
        'shape': '(day_len,num_stock)',
        'description': '当日成交额'
    },
    'M_O': {
        'shape': '(day_len,num_stock,minute_len)',
        'description': '日内每一分钟的开盘价'
    },
    'M_C': {
        'shape': '(day_len,num_stock,minute_len)',
        'description': '日内每一分钟的收盘价'
    },
    'M_H': {
        'shape': '(day_len,num_stock,minute_len)',
        'description': '日内每一分钟的最高价'
    },
    'M_L': {
        'shape': '(day_len,num_stock,minute_len)',
        'description': '日内每一分钟的最低价'
    },
    'M_V': {
        'shape': '(day_len,num_stock,minute_len)',
        'description': '日内每一分钟的成交额'
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
    '原则': '将算子按照输入和输出的变量类型以及混频同频等格式作区分，总共有五个层级，分别是纲目科属种。纲有两类，表示算子的使用领域；目有四类，表示算子的输出类型；科有两类，表示算子使用的变量类型频度；属的类别较多，表示算子输入的变量组合。种对应到单个算子。这种分类是一种无交叉的分类，每个算子只可能属于一个类别',
    '具体划分': {
        '纲': {
            '通用纲': '该纲中的算子是用途广泛的数据操作动作，可以用于几乎所有变量类型，一般用于生成专用纲算子中的传递变量；Others.py中的算子均为通用纲算子',
            '专用纲': '该纲中的算子专门用于生成TypeABCD四种变量类型，专用纲算子均在ToA.py，ToB.py，ToC.py，ToD.py中。它包含的算子类型多样，需要继续用目科属层级进行划分'
        },
        '目': {
            'A目': '输出类型为TypeA',
            'B目': '输出类型为TypeB',
            'C目': '输出类型为TypeC',
            'D目': '输出类型为TypeD'
        },
        '科': {
            '同频科': '输入的 变量类型组合和输出的变量类型 全部都属于日频或者分钟频中的同一个频度',
            '混频科': '输入的 变量类型组合和输出的变量类型 既包含日频，也包含分钟频'
        },
        '属': {
            '属的划分依据是输入的变量类型组合，如BBD属表示输入的变量类型有三个，分别是TypeB,TypeB 和 TypeD；A属表示输入的变量类型只有TypeA'
        },
    },
}

Abstraction_Action_Group = {
    'name': {'action;抽象动作分类:根据算子是否改变算子特定维度中的rank进行分类'
             },
    'rule': {
        '共轭:conjugate,不改变rank，如M_cs_scale',
        '扭曲:twist,产生了易懂直观的结构性变化，如edge_flip',
        '混沌:chaos,四种基本运算及其延伸与组合',
        '自适应:self-adjust,在特定维度的rank变化受自身特性影响，如D_ts_to_max'
        '剥离:strip,回归类的算子，剥离出数据的某些特征'
        '描述:describe,包含统计操作，如求均值，方差等'
        '提纯:purify，包含mask掩码操作'
    },
    'example': {}
}

op_info = {
    'D_at_abs': {
        'description': '输入一个A类型的张量x，计算并返回其绝对值。此操作用于消除负号的影响，保留数值的大小。',
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
    'D_cs_rank': {
        'description': '输入一个A类型的张量x，计算其截面分位数。首先去除NaN值，然后对数据进行排序并计算分位数。此操作用于将数据标准化到[0, 1]区间，便于后续分析。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'A属'
            },
            'action': {
                '描述'
            }
        }
    },
    'D_cs_scale': {
        'description': '输入一个A类型的张量x，计算其截面最大最小值标准化。通过将数据缩放到[0, 1]区间并加1，使得数据范围为[1, 2]。此操作用于标准化数据，消除量纲差异。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'A属'
            },
            'action': {
                '描述'
            }
        }
    },
    'D_cs_zscore': {
        'description': '输入一个A类型的张量x，计算其z-score标准化。此操作用于将数据标准化到均值为0、标准差为1的分布。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'A属'
            },
            'action': {
                '描述'
            }
        }
    },
    'D_cs_harmonic_mean': {
        'description': '输入一个A类型的张量x，计算其调和平均值。此操作用于平滑数据中的极端值。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'A属'
            },
            'action': {
                '描述'
            }
        }
    },
    'D_cs_demean': {
        'description': '输入一个A类型的张量x，计算其去均值后的绝对值。此操作用于消除数据的中心趋势。',
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
    'D_cs_winsor': {
        'description': '输入一个A类型的张量x，计算其Winsor处理后的结果。此操作用于限制数据的极端值，避免异常值对分析的影响。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'A属'
            },
            'action': {
                '剥离'
            }
        }
    },
    'D_cs_demean_industry': {
        'description': '输入一个A类型的张量x和一个E类型的行业哑变量矩阵y。计算每个行业的均值，并对每个股票的日频数据进行去均值处理，返回绝对值。',
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
        'description': '输入一个A类型的张量x和一个E类型的行业哑变量矩阵y。对每个股票的日频数据进行行业中性化处理。',
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
        'description': '输入一个A类型的张量x和一个G类型的阈值thresh。对x的分位数处于阈值以外的部分取负号。此操作用于突出数据的极端值。',
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
                '描述', '自适应'
            }
        }
    },

    'D_ts_max': {
        'description': '输入一个A类型的张量x和一个F类型的int参数lookback。计算x在回溯lookback天内的最大值，用于识别时间序列中的峰值。此操作可用于识别股票价格的短期高点。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            },
            'action': {
                '描述', '自适应'
            }
        }
    },
    'D_ts_min': {
        'description': '输入一个A类型的张量x和一个F类型的int参数lookback。计算x在回溯lookback天内的最小值，用于识别时间序列中的谷值。此操作可用于识别股票价格的短期低点。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '同频科',
                '属': 'AF属'
            },
            'action': {
                '描述', '自适应'
            }
        }
    },
    'D_ts_delay': {
        'description': '输入一个A类型的张量x和一个F类型的int参数d。对输入张量进行d天的延迟处理，常用于构建滞后变量，动量或反转策略。',
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
        'description': '输入一个A类型的张量x和一个F类型的int参数d。计算输入张量与d天前的差值，用于衡量股票价格的短期波动。',
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
        'description': '输入一个A类型的张量x和一个F类型的int参数lookback。计算输入张量在回溯lookback天内的涨跌幅，用于构建动量指标或分析股票价格的短期表现。',
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
        'description': '输入一个A类型的张量x和一个F类型的int参数lookback。计算输入张量在回溯lookback天内的均值，用于分析股票价格的短期趋势，辅助移动平均线策略。',
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
        'description': '输入一个A类型的张量x和一个F类型的int参数lookback。计算输入张量在回溯lookback天内的调和平均值，用于平滑时间序列中的极端值。可用于分析股票价格的短期趋势，尤其在处理极端波动时更为有效。',
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
        'description': '输入一个A类型的张量x和一个F类型的int参数lookback。计算输入张量在回溯lookback天内的标准差，用于评估时间序列的波动性。可用于分析股票价格的短期波动性。',
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
        'description': '输入一个A类型的张量x和一个F类型的int参数d。计算x与x回溯d天内的最大值的比值。可用于分析股票价格相对于短期高点的表现。',
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
        'description': '输入一个A类型的张量x和一个F类型的int参数d。计算x与x回溯d天内的最小值的比值。可用于分析股票价格相对于短期低点的表现。',
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
        'description': '输入一个A类型的张量x和一个F类型的int参数d。计算x与x回溯d天内的均值的比值。可用于分析股票价格相对于短期均值的表现。',
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
        'description': '输入一个A类型的张量x和一个F类型的int参数d。计算x与x回溯d天内的最大值与最小值的差值的比值。可用于分析股票价格的短期波动范围。',
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
        'description': '输入一个A类型的张量x和一个F类型的int参数d。去除x在回溯d天内的趋势，返回残差。此操作可用于分析股票价格的非趋势性波动。',
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

    # OP_BD2A 类中的算子
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
        'description': '输入B类型的张量x和D类型的mask。对mask为True的部分计算x的标准差。此操作可用于分析特定分钟区间内的股票价格波动性，辅助日内交易策略或分析日内风险。',
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
        'description': '输入B类型的张量x和D类型的mask。对mask为True的部分计算x的和。此操作可用于分析特定分钟区间内的股票价格总和，辅助日内交易策略或分析特定事件的影响。',
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
        'description': '输入B类型的张量x和D类型的mask。对mask为True的部分计算x的乘积。此操作可用于分析特定分钟区间内的股票价格乘积，辅助构建复合指标或分析特定事件的影响。',
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

    # OP_BBD2A 类中的算子
    'D_Minute_area_weight_mean': {
        'description': '输入一个B类型的张量x，一个权重张量weight，以及一个D类型的掩码mask。对mask为True的部分，以weight为权重计算x的加权均值。此操作可用于分析特定分钟区间内的股票价格加权均值，辅助日内交易策略或分析日内波动。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',
                '属': 'BBD属'
            },
            'action': {
                '描述', '提纯'
            }
        }
    },
    'D_Minute_area_corr': {
        'description': '输入两个B类型的张量x和y，以及一个D类型的掩码mask。对mask为True的部分计算x和y的相关性。此操作可用于分析特定分钟区间内的股票价格相关性。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',
                '属': 'BBD属'
            },
            'action': {
                '描述', '提纯'
            }
        }
    },
    'D_Minute_area_rankcorr': {
        'description': '输入两个B类型的张量x和y，以及一个D类型的掩码mask。对mask为True的部分计算x和y的秩相关性。此操作可用于分析特定分钟区间内的股票价格秩相关性。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',
                '属': 'BBD属'
            },
            'action': {
                '描述', '提纯'
            }
        }
    },
    'D_Minute_area_bifurcate_mean': {
        'description': '输入两个B类型的张量x和y，以及一个D类型的掩码mask。计算y在mask区域的均值,根据该均值将mask分成两部分，''记为mask1(mask为true 且 y大于该均值)和mask2(mask为true 且 y小于该均值)，分别计算x在mask1 和mask2的均值， 再输出两个均值的差。该算子意在以y在mask部分的值为基准，将mask分成两部分，再将x在这两部分的均值做对冲。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',
                '属': 'BBD属'
            },
            'action': {
                '描述'
            }
        }
    },
    'D_Minute_area_bifurcate_std': {
        'description': '输入两个B类型的张量x和y，以及一个D类型的掩码mask。计算x在分叉区域的标准差。此操作可用于分析特定分钟区间内的股票价格波动性差异，辅助日内交易策略或分析日内波动。',
        'classification': {
            'interface': {
                '目': 'A目',
                '科': '混频科',
                '属': 'BBD属'
            },
            'action': {
                '描述'
            }
        }
    },

    # OP_BB2A 类中的算子
    'D_Minute_corr': {
        'description': '输入两个B类型的张量x和y。计算x和y在分钟频率上的相关性。',
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
        'description': '输入一个B类型的张量x和一个权重张量weight。计算x的加权平均值。',
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


    'D_Minute_abnormal_point_count': {
        'description': '输入D类型的mask。计算mask中标记为True的数量。此操作可用于分析特定分钟区间内的异常点数量',
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
    }
,
    'M_ignore_wobble': {
        'description': '输入一个B类型的张量x，将开盘前5分钟和收盘前5分钟的数据设置为NaN。此操作用于去除开盘和收盘时的异常波动。',
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
        'description': '输入一个B类型的张量x，计算每个股票每天的z-score标准化。此操作用于将数据标准化到均值为0、标准差为1的分布。',
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
        'description': '输入一个B类型的张量x，计算每个股票每天的分位数排名。此操作用于将数据标准化到[0, 1]区间',
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
        'description': '输入一个B类型的张量x，计算每个股票每天的最大最小值标准化。此操作用于将数据缩放到[1, 2]区间，消除量纲差异。',
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
    'M_at_sign': {
        'description': '输入一个B类型的张量x，计算其符号函数，返回-1、0或1。此操作用于提取数据的符号信息。',
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
    'M_cs_demean': {
        'description': '输入一个B类型的张量x，计算每个股票每天的数据与均值的绝对差。此操作用于衡量数据偏离均值的程度。',
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
    'M_cs_winsor': {
        'description': '输入一个B类型的张量x，对数据进行Winsor处理，将分位数小于0.05或大于0.95的部分替换为相应分位数的值。此操作用于限制极端值的影响。',
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
        'description': '输入一个B类型的张量x，计算其绝对值。此操作用于消除负号的影响，保留数值的大小。',
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

    # OP_BB2B 类中的算子
    'M_at_add': {
        'description': '输入两个B类型的张量x和y，逐元素相加。此操作用于合并两个张量的数值。',
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
        'description': '输入两个B类型的张量x和y，逐元素相减。此操作用于计算两个张量之间的差异。',
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
        'description': '输入两个B类型的张量x和y，逐元素相除。当y为0时，结果为NaN。此操作用于计算两个张量的比值。',
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
    'M_cs_cut': {
        'description': '输入两个B类型的张量x和y，根据x的符号对y进行截断处理。x为正时保留y的值，x为负时将y置为0。此操作用于根据x的符号调整y的值。',
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
        'description': '输入两个B类型的张量x和y，计算(x - mean(y)) * y。此操作用于衡量x相对于y均值的偏离程度。',
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
        'description': '输入两个B类型的张量x和y，逐元素相乘。此操作用于计算两个张量的乘积。',
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
        'description': '输入两个B类型的张量x和y，计算(x - y) / (|x| + |y|)。此操作用于衡量两个变量之间的相对差异，避免因绝对值大小不同而导致的偏差。',
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

    # OP_BA2B 类中的算子
    'M_toD_standard': {
        'description': '输入一个B类型的张量M_tensor和一个A类型的张量D_tensor。将M_tensor除以D_tensor的转置并扩展维度后的结果。此操作用于将分钟数据标准化到日频数据的尺度。',
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

    # OP_BG2B 类中的算子
    'M_cs_edge_flip': {
        'description': '输入一个B类型的张量x和一个G类型的阈值thresh。根据阈值对x的边缘部分进行翻转处理。此操作用于突出数据的极端值。',
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

    # OP_BF2B 类中的算子
    'M_ts_delay': {
        'description': '输入一个B类型的张量x和一个F类型的int参数d。对输入张量进行d分钟的延迟处理，常用于构建滞后变量。',
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
        'description': '输入一个B类型的张量x和一个F类型的int参数lookback。计算x在回溯lookback分钟内的涨跌幅。此操作用于分析时间序列的相对变化。',
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
    'M_ts_delta': {
        'description': '输入一个B类型的张量x和一个F类型的int参数lookback。计算x与回溯lookback分钟前的差值。此操作用于衡量时间序列的变化幅度。',
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
        'description': '输入一个B类型的张量x和一个F类型的int参数neighbor_range。计算每个时间点左侧邻居范围内的均值。此操作用于平滑股票价格时间序列，便于分析短期趋势。',
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
        'description': '输入一个B类型的张量x和一个F类型的int参数neighbor_range。计算每个时间点中间邻居范围内的均值。此操作用于平滑股票价格时间序列，便于分析中期趋势。',
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
        'description': '输入一个B类型的张量x和一个F类型的int参数neighbor_range。计算每个时间点右侧邻居范围内的均值。此操作用于平滑股票价格时间序列，便于分析未来趋势。',
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
        'description': '输入一个B类型的张量x和一个F类型的int参数neighbor_range。计算每个时间点左侧邻居范围内的标准差。此操作用于评估股票价格的短期波动性，便于分析市场风险。',
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
        'description': '输入一个B类型的张量x和一个F类型的int参数neighbor_range。计算每个时间点中间邻居范围内的标准差。此操作用于评估股票价格的中期波动性，便于分析市场风险。',
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
        'description': '输入一个B类型的张量x和一个F类型的int参数neighbor_range。计算每个时间点右侧邻居范围内的标准差。此操作用于评估股票价格的未来波动性，便于分析市场风险。',
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
        'description': '输入一个B类型的张量x和一个F类型的int参数neighbor_range。计算每个时间点左侧邻居范围内的乘积。此操作用于计算股票价格的累积效应，便于分析短期趋势的强度。',
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
    'M_ts_product_mid_neighbor': {
        'description': '输入一个B类型的张量x和一个F类型的int参数neighbor_range。计算每个时间点中间邻居范围内的乘积。此操作用于计算股票价格的累积效应，便于分析中期趋势的强度。',
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
    'M_ts_product_right_neighbor': {
        'description': '输入一个B类型的张量x和一个F类型的int参数neighbor_range。计算每个时间点右侧邻居范围内的乘积。此操作用于计算股票价格的累积效应，便于分析未来趋势的强度。',
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
    'Dmask_min': {
        'description': '输入一个A类型的张量x和一个F类型的int参数lookback。展开过去lookback天的数据，标记最小的1/4天为True，其余为False。此操作用于识别股票价格的低谷区域，便于分析市场底部或超卖状态。',
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
        'description': '输入一个A类型的张量x和一个F类型的int参数lookback。展开过去lookback天的数据，标记最大的1/4天为True，其余为False。此操作用于识别股票价格的高峰区域，便于分析市场顶部或超买状态。',
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
        'description': '输入一个A类型的张量x和一个F类型的int参数lookback。展开过去lookback天的数据，标记中间的1/2天为True，其余为False。此操作用于识别股票价格的中性区域，便于分析市场平衡状态或过滤极端波动。',
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
        'description': '输入一个A类型的张量x和一个F类型的int参数lookback。展开过去lookback天的数据，进行标准化处理，标记大于均值+标准差的部分为True。此操作用于识别股票价格的高波动区域，便于分析市场异常上涨或动量信号。',
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
        'description': '输入一个A类型的张量x和一个F类型的int参数lookback。展开过去lookback天的数据，进行标准化处理，标记小于均值-标准差的部分为True。此操作用于识别股票价格的低波动区域，便于分析市场异常下跌或价值信号。',
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
        'description': '输入一个B类型的张量x，返回日内的最小1/4部分作为掩码。此操作用于识别日内价格的低谷区域，便于分析市场底部或超卖状态。',
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
        'description': '输入一个B类型的张量x，返回日内的最大1/4部分作为掩码。此操作用于识别日内价格的高峰区域，便于分析市场顶部或超买状态。',
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
        'description': '输入一个B类型的张量x，返回日内的中间1/2部分作为掩码。此操作用于识别日内价格的中性区域，便于分析市场平衡状态或过滤极端波动。',
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
        'description': '输入一个B类型的张量x，返回日内最大值和最小值之间的部分作为掩码。此操作用于识别日内价格的主要波动区间，便于分析价格波动范围。',
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
        'description': '输入一个B类型的张量x，返回大于均值加1倍标准差的部分作为掩码。此操作用于识别日内价格的高波动区域，便于分析市场异常上涨或动量信号。',
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
        'description': '输入一个B类型的张量x，返回小于均值减1倍标准差的部分作为掩码。此操作用于识别日内价格的低波动区域，便于分析市场异常下跌或价值信号。',
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
        'description': '输入一个B类型的张量x，返回开盘后的第一个小时作为掩码。此操作用于分析开盘初期的价格行为，便于捕捉市场开盘时的动量或反转信号。',
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
        'description': '输入一个B类型的张量x，返回收盘前的第一个小时作为掩码。此操作用于分析收盘前的价格行为，便于捕捉市场收盘时的动量或反转信号。',
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
        'description': '输入一个B类型的张量x，返回中间的两个小时作为掩码。此操作用于分析日内价格的中期行为，便于捕捉市场的主要波动区间。',
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
        'description': '输入一个B类型的张量x，返回早上的两个小时作为掩码。此操作用于分析早盘的价格行为，便于捕捉市场开盘后的趋势或反转信号。',
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
        'description': '输入一个B类型的张量x，返回下午的两个小时作为掩码。此操作用于分析午盘的价格行为，便于捕捉市场收盘前的趋势或反转信号。',
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

    # OP_BA2D 类中的算子
    'Mmask_day_plus': {
        'description': '输入一个B类型的张量m_tensor和一个A类型的张量d_tensor，返回大于日频数据的部分作为掩码。此操作用于识别日内价格高于日均值的时段，便于分析日内动量信号。',
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
        'description': '输入一个B类型的张量m_tensor和一个A类型的张量d_tensor，返回小于日频数据的部分作为掩码。此操作用于识别日内价格低于日均值的时段，便于分析日内价值信号。',
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

    # OP_BF2D 类中的算子
    'Mmask_rolling_plus': {
        'description': '输入一个B类型的张量x和一个F类型的int参数lookback，返回大于lookback期内最大的日平均较大值的部分作为掩码。此操作用于识别滚动窗口内价格高于历史高位的时段，便于分析长期动量信号。',
        'classification': {
            'interface': {
                '目': 'D目',
                '科': '混频科',
                '属': 'BF属'
            },
            'action': {
                '描述', '提纯'
            }
        }
    },
    'Mmask_rolling_sub': {
        'description': '输入一个B类型的张量x和一个F类型的int参数lookback，返回小于lookback期内最小的日平均较小值的部分作为掩码。此操作用于识别滚动窗口内价格低于历史低位的时段，便于分析长期价值信号。',
        'classification': {
            'interface': {
                '目': 'D目',
                '科': '混频科',
                '属': 'BF属'
            },
            'action': {
                '描述', '提纯'
            }
        }
    },

    # OP_DD2D 类中的算子
    'Mmask_and': {
        'description': '输入两个D类型的掩码m_mask_x和m_mask_y，返回两个掩码的逻辑与结果。此操作用于结合多个掩码条件，便于分析复合市场信号。',
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
        'description': '输入两个D类型的掩码m_mask_x和m_mask_y，返回两个掩码的逻辑或结果。此操作用于合并多个掩码条件，便于分析复合市场信号。',
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


