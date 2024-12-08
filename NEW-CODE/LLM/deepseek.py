import json
from openai import OpenAI
from data_info import data_info
from op_info import op_info

api_key = ''

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

summary = '''
换手率因子已经体现出了较强的选股能力，但仍然存在波动率较大的问题。由于换手率和个股的市值有较强的相关性，换手率的变化率相对换手率更能体现出因子的个性。采用日内振幅作为权重对换手率的变化率做加权平均，让量的变化有价格变化的确认。
'''

system_prompt = (f'''
你是一个金融方向的研究助手，你将得到一份有关金融市场的研报的总结，请你从这份研报总结进行分析并从使用数据、因子表达式、因子表达式分解、因子逻辑四个角度进行股票市场因子的生成。
你可以使用的数据及其格式与描述为：
{data_info}，
你可以使用的算子及其输入、输出格式与功能描述为：
{op_info}，
示例输入：
研报总结：{summary}
示例JSON输出：
{{
    {{
        "数据":"open, high, low, close, volume",
        "因子表达式":"D_ts_weight_mean(D_at_div(volume,D_ts_mean(volume,20)),D_at_div(high,low),20)",
        "因子表达式分解":{{
            "代理变量":"D_at_div(volume,D_ts_mean(volume,20))，相对于20日均值的换手率",
            "代表性子集":"无",
            "统计量":"weight_mean"
        }},
        "因子逻辑":"换手率因子已经体现出了较强的选股能力，但仍然存在波动率较大的问题。\
        由于换手率和个股的市值有较强的相关性，换手率的变化率相对换手率更能体现出因子的个性。\
        采用日内振幅作为权重对换手率的变化率做加权平均，让量的变化有价格变化的确认。"
    }}
}}
''')

user_prompt = f'''
研报总结：换手率因子已经体现出了较强的选股能力，但仍然存在波动率较大的问题。由于换手率和个股的市值有较强的相关性，换手率的变化率相对换手率更能体现出因子的个性。采用日内振幅作为权重对换手率的变化率做加权平均，让量的变化有价格变化的确认。
'''


messages = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]


response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    response_format={
        'type': 'json_object'
    }
)

print(json.loads(response.choices[0].message.content))