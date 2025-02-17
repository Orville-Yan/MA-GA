from pathlib import Path
import sys
parent_path = Path(__file__).resolve().parent.parent
sys.path.append(r'{}'.format(parent_path))
from Info import *
import json

paper_example = '''
换手率因子已经体现出了较强的选股能力，但仍然存在波动率较大的问题。由于换手率和个股的市值有较强的相关性，换手率的变化率相对换手率更能体现出因子的个性。采用日内振幅作为权重对换手率的变化率做加权平均，让量的变化有价格变化的确认。
'''

with open('proxy.json', 'r') as file:
    proxy_example = json.load(file)

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
首先判断研报有没有代理变量，如果没有，则放弃摘要；如果有，
示例JSON输出为：{proxy_example}
用户输入如下，请根据输入进行回答
'''
)

user_prompt = '''
第{x}份研报总结：{summary}
'''