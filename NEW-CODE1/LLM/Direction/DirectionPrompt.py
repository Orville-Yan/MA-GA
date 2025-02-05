from langchain.prompts import PromptTemplate,FewShotPromptTemplate

Visual_Prompt = {
    'context':"你要分析这个表达式的含义，请不要输出多余的形式内容。一个表达式由算子和算符两部分构成。算符是含ARG的，算子会出现在()左侧。表达式、算子的含义和算符的含义都会由用户输入。",
    'instruction':"请你考虑每一分钟或者每一天的因子值的含义，以及数值比较大说明了什么。",
}

examples = [
    {
        "input": "当表达式为M_cs_umr(M_cs_scale(M_ts_delay(M_V, 60)), M_cs_scale(M_ts_mean_left_neighbor(M_V, 5)))，请你分析因子值表达大或者比较小所表达的含义。",
        "output": "M_cs_umr有两个输入。M_cs_scale(M_ts_delay(M_V, 60))表示当日分钟成交量标准化以后 60分钟前的分位数结果；M_cs_scale(M_ts_mean_left_neighbor(M_V, 5))表示前五分钟成交量的均值标准化以后的结果。M_cs_umr将这二者结合起来，M_cs_umr是(x - mean(y)) * y，而mean(y)约等于0.5，于是这个表达式按照x的分位数和0.5的关系 对y做了加权。若某一分钟的因子值比较大，有两种情况。一种是y为负值，同时x分位数很小，这样才会造成(x - mean(y))小于零且绝对值很大，这样一来二者乘积才会比较大。第二种是y为正值，同时x也很大。由于M_cs_scale(M_ts_mean_left_neighbor(M_V, 5))是y，它恒大于零，所以我们得到的是第二种情况，再考虑x和y的具体含义，我们就能得到该表达式下，某一分钟的因子值比较大所表达的含义：60分钟前这一分钟的成交量很大，并且前面的最近5分钟平均成交量也非常大。因子值比较小则相反，由于x和y均为非负，于是因子值比较小说明60分钟前这一分钟的成交量非常小并且最近五分钟的平均成交量也很小。"
    },
    {
        "input": "当表达式为M_at_sub(M_cs_rank(M_ts_mean_left_neighbor(M_V, 60)), M_cs_zscore(M_ts_mean_left_neighbor(M_V, 10)))，请你分析因子值表达大或者比较小所表达的含义。",
        "output": "M_cs_rank(M_ts_mean_left_neighbor(M_V, 60))表示过去一小时成交量均值的分位数，M_cs_zscore(M_ts_mean_left_neighbor(M_V, 10))则表示过去10分钟成交量的均值标准化以后的结果。M_at_sub是x-y。若因子值比较大，说明过去十分钟成交量相对缩量；若因子值比较小，说明过去十分钟出现了放量。"
    },
]


def few_shot_examples():

    example_template = """
    输入: {input}
    输出: {output}
    """

    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template=example_template
    )

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="以下是几个帮助你分析因子表达式含义的示例，主要是分析某一时刻因子值比较大或者比较小说明了什么。",
        suffix="输入: {user_input}\n输出:",
        input_variables=["user_input"],
        example_separator="\n\n"
    )

    return few_shot_prompt
