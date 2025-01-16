OP_extraction_Prompt=('在该任务中不要添加任何额外的描述或解释,只需要输出一个字典的str格式，使我能够直接转换成字典格式。我会给你一个因子表达式{factor_str}，包含算子和变量两部分。你现在有一个算子池的列表{op_info}和他们的{classification},还有一个{variable_info}。你需要把因子表达式中涉及到的算子信息从op_info中提取出来，如果是变量，就忽略不做输出。关于算子输出的格式，你必须要和算子池的列表一致(以字典的字符串形式)，你的工作只是提取，不需要对里面的内容所任何改变')

Specific_Prompt = {
    'seed':('这里的动作是在去除数据的量纲')
}

Visual_Prompt = ('我给你一个动作，你要解释这个动作产生的效果。'+
                 '这个动作{actions}由变量代称和算子构成，算子的信息在{op_info}中，变量代称所指代的实际变量在变量存储器{arg_memorizer}中，这些变量的含义在含义存储器{meaning_memorizer}'+
                 '{specific}')

