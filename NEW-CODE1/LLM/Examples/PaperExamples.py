from pathlib import Path
import sys
parent_path = Path(__file__).resolve().parent.parent
sys.path.append(r'{}'.format(parent_path))
from Info import *
import json

paper_example = '''
换手率因子已经体现出了较强的选股能力，但仍然存在波动率较大的问题。由于换手率和个股的市值有较强的相关性，换手率的变化率相对换手率更能体现出因子的个性。采用日内振幅作为权重对换手率的变化率做加权平均，让量的变化有价格变化的确认。
'''

system_prompt = (
    f'''
你是一个金融方向的研究助手，你将得到多份有关金融市场的研报的总结，请你从这份研报总结进行分析并从使用数据、因子表达式、因子表达式分解、因子逻辑四个角度进行股票市场因子的生成。
你可以使用的数据及其格式与描述为：
{data_info}，
数据对应的数据类型为：
{VariableType_info}，

你可以使用的算子规则及其输入、输出格式与功能描述为：
算子规则：{Interface_Protocol_Group}
算子描述：{op_info}，
要求因子表达式最后输出为TypeA，也即输出频率为日频。

示例输入：
第1份研报总结：{paper_example}，
第2份研报总结：{paper_example}
示例JSON输出：
{{"第1份研报分析":
    {{
        "数据":"D_O, D_H, D_L, D_C, D_V",
        "因子表达式":"D_ts_weight_mean(D_at_div(D_V,D_ts_mean(D_V,20)),D_at_div(D_H,D_L),20)",
        "因子表达式分解":{{
            "代理变量":"D_at_div(D_V,D_ts_mean(D_V,20))，相对于20日均值的换手率",
            "代表性子集":"无",
            "统计量":"weight_mean"
        }},
        "因子逻辑":"换手率因子已经体现出了较强的选股能力，但仍然存在波动率较大的问题。\
        由于换手率和个股的市值有较强的相关性，换手率的变化率相对换手率更能体现出因子的个性。\
        采用日内振幅作为权重对换手率的变化率做加权平均，让量的变化有价格变化的确认。"
    }},
  "第2份研报分析":
    {{
        "数据":"D_O, D_H, D_L, D_C, D_V",
        "因子表达式":"D_ts_weight_mean(D_at_div(D_V,D_ts_mean(D_V,20)),D_at_div(D_H,D_L),20)",
        "因子表达式分解":{{
            "代理变量":"D_at_div(D_V,D_ts_mean(D_V,20))，相对于20日均值的换手率",
            "代表性子集":"无",
            "统计量":"weight_mean"
        }},
        "因子逻辑":"换手率因子已经体现出了较强的选股能力，但仍然存在波动率较大的问题。\
        由于换手率和个股的市值有较强的相关性，换手率的变化率相对换手率更能体现出因子的个性。\
        采用日内振幅作为权重对换手率的变化率做加权平均，让量的变化有价格变化的确认。"
    }}
}}
用户输入如下，请根据输入进行回答
'''
)

user_prompt = '''
第{x}份研报总结：{summary}
'''