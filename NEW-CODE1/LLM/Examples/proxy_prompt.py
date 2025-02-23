from pathlib import Path
import sys
parent_path = Path(__file__).resolve().parent.parent
sys.path.append(r'{}'.format(parent_path))

from OP.Info import *
import json

paper_example = '''
换手率因子已经体现出了较强的选股能力，但仍然存在波动率较大的问题。由于换手率和个股的市值有较强的相关性，换手率的变化率相对换手率更能体现出因子的个性。采用日内振幅作为权重对换手率的变化率做加权平均，让量的变化有价格变化的确认。
'''

with open('proxy.json', 'r') as file:
    proxy_example = json.load(file)
    proxy_example_str = json.dumps(proxy_example, indent=2, ensure_ascii=False).replace('{', '{{').replace('}', '}}')

def escape_dict(d):
    return json.dumps(d, indent=2, ensure_ascii=False, default=str) \
             .replace("{", "{{") \
             .replace("}", "}}")

data_info_str = escape_dict(data_info)
VariableType_info_str = escape_dict(VariableType_info)
Interface_Protocol_Group_str = escape_dict(Interface_Protocol_Group)
op_info_str = escape_dict(op_info)

system_prompt = (
    f'''
    
您作为专业金融量化因子挖掘引擎，请严格按以下框架从研报摘要中提取代理变量信息：




要求因子表达式最后输出为TypeA，也即输出频率为日频。

# 处理流程
1. 逐份分析输入内容，判断每份研报是否包含可量化的代理变量（满足以下任一条件）：
   - 明确提及可观测的市场行为指标
   - 描述可测量的投资者行为模式
   - 提出可转换为数学表达式的市场特征
   
   若不符合则直接丢弃该篇摘要

2. 对有效研报执行：
   a) 提取【宏观信息】：用"XXX是否YYY"句式归纳核心命题
   b) 识别【代理变量】：用动宾结构描述量化观测点
   c) 构建【代理变量表达式】：
      - 严格使用指定数据字段：{data_info_str}
      - 仅使用核准算子：{op_info_str}
      - 数据对应的数据类型为：{VariableType_info_str}
      - 算子规则为：{Interface_Protocol_Group_str}
      - 确保输出频率为日频（TypeA）
   d) 生成【近义替换】：提供2种参数/算子级替代方案，保持数学逻辑一致

# 输出规范
要求严格遵循此JSON结构：
    "宏观信息": "命题判断句式",
    "代理变量": "动词开头的量化观测描述",
    "代理变量表达式": "嵌套算子表达式", 
    "近义替换": [
      "参数替换方案（如MC→MV）",
      "算子替换方案（如std→pctchg）"
    ]

# 重点约束
1. 表达式必须使用M/D_前缀算子，如：M_ts_pctchg(MC,5)
2. 最终输出类型必须为TypeA（日频）
3. 禁止返回非JSON内容，拒绝Markdown格式

示例输入：
第1份研报总结：{paper_example}，
第2份研报总结：{paper_example}
首先判断研报有没有代理变量，如果没有，则放弃摘要；如果有，
示例JSON输出为：{proxy_example_str},回答中不能包括例子中的内容
用户输入如下，请根据输入进行回答，你只能返回严格的JSON格式内容
'''
)

user_prompt = '''
第{x}份研报总结：{summary}
'''