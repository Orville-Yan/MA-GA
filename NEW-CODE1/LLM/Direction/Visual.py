from RPN import *
from LLM.Info import *
from DirectionPrompt import *
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langgraph.graph import  StateGraph,START
from langchain_community.llms.moonshot import Moonshot

LLM = Moonshot(model="moonshot-v1-8k", api_key='sk-MpplKWdi4h0RaEkTm405LcAZWbBEIr9AaiJ4XjJr7QYecRNa')

class Visual:
    def __init__(self, rpn: str):
        self.factor = rpn

    def get_initial_state(self):

        parser = RPN_Parser(self.factor)
        parser.get_tree_structure()
        parser.parse_tree()
        parsed_rpn = parser.tree2dict()
        self.tree = parsed_rpn

        self.initial_state = {
            'abbreviation_mode': parsed_rpn,
            'initial_arg_memorizer': {
                'ARG0': 'D_O',
                'ARG1': 'D_C',
                'ARG2': 'D_H',
                'ARG3': 'D_L',
                'ARG4': 'D_V',
                'ARG5': 'M_O',
                'ARG6': 'M_C',
                'ARG7': 'M_H',
                'ARG8': 'M_L',
                'ARG9': 'M_V'
            },
            'initial_meaning_memorizer': {
                'ARG0': '当日开盘价',
                'ARG1': '当日收盘价',
                'ARG2': '当日最高价',
                'ARG3': '当日最低价',
                'ARG4': '当日成交额',
                'ARG5': '日内每一分钟的开盘价',
                'ARG6': '日内每一分钟的收盘价',
                'ARG7': '日内每一分钟的最高价',
                'ARG8': '日内每一分钟的最低价',
                'ARG9': '日内每一分钟的成交额'
            },
            'seed_arg_memorizer': {},
            'seed_meaning_memorizer': {},
            'root_arg_memorizer': {},
            'root_meaning_memorizer': {},
            'branch_arg_memorizer': {},
            'branch_meaning_memorizer': {},
            'trunk_arg_memorizer': {},
            'trunk_meaning_memorizer': {},
            'subtree_arg_memorizer': {},
            'subtree_meaning_memorizer': {},
            'tree_arg_memorizer': {},
            'tree_meaning_memorizer': {},
        }

    def extract_memorizer(self, organ_abbreviation, state):
        arg_memorizer = {}
        meanning_memorizer = {}

        position = []
        for index, char in enumerate(organ_abbreviation):
            if char in ['(', ')', ',']:
                position.append(int(index))

        arg = []
        for i in range(len(position) - 1):
            s = organ_abbreviation[position[i] + 1:position[i + 1]].strip()
            if not s.isdigit():
                arg.append(s)

        for s in arg:
            if s[-2:].isdigit():
                flag = 5
            else:
                flag = 4

            if len(s[:-flag]) == 0:
                arg_memorizer[s] = state['initial_arg_memorizer'][s]
                meanning_memorizer[s] = state['initial_meaning_memorizer'][s]
            else:
                arg_memorizer[s[-flag:]] = state[f'{s[:-flag]}arg_memorizer'][s[-flag:]]
                meanning_memorizer[s[-flag:]] = state[f'{s[:-flag]}meaning_memorizer'][s[-flag:]]

        return arg_memorizer, meanning_memorizer

    def abbrev_analysis(self, organ_abbreviation, op, arg_memorizer, meaning_memorizer):
        system_message1 = SystemMessagePromptTemplate.from_template(
            Visual_Prompt['context'])
        system_message2 = SystemMessagePromptTemplate.from_template(
            Visual_Prompt['instruction'])
        user_message = HumanMessagePromptTemplate.from_template(
            f"表达式为{organ_abbreviation},ARG代表的变量为{arg_memorizer}。算子的具体含义为{op},ARG代表的变量含义为{meaning_memorizer}。请你结合ARG代表的变量含义和算子含义，分析表达式的含义。")

        few_shot_prompt_str = few_shot_examples().format(user_input="")
        chat_prompt = ChatPromptTemplate.from_messages([
            system_message1,
            system_message2,
            HumanMessagePromptTemplate.from_template(few_shot_prompt_str),
            user_message
        ])

        chain = chat_prompt | LLM
        response = chain.invoke({'organ_abbreviation': organ_abbreviation,
                                 'arg_memorizer': arg_memorizer,
                                 'op': op,
                                 'meaning_memorizer': meaning_memorizer,
                                 })
        return response

    def generate_seperate_node(self, organs):

        def node_func(state):

            for organ in organs:

                for node_i in range(len(state['abbreviation_mode'][organ])):

                    organ_abbreviation = state['abbreviation_mode'][organ][node_i]

                    state[f'{organ}_arg_memorizer'][f'ARG{node_i}'] = organ_abbreviation

                    used_op_info = extract_op(expression=organ_abbreviation)

                    arg_memorizer, meaning_memorizer = self.extract_memorizer(organ_abbreviation, state)

                    state[f'{organ}_meaning_memorizer'][f'ARG{node_i}'] = self.abbrev_analysis(
                        organ_abbreviation=organ_abbreviation,
                        op=dict2str(used_op_info),
                        arg_memorizer=dict2str(arg_memorizer),
                        meaning_memorizer=dict2str(meaning_memorizer),
                    )

                return state

        return node_func

    def generate_graph(self):

        self.graph = StateGraph(state_schema=dict)

        for organs in [['seed'], ['root', 'branch'], ['trunk'], ['subtree'], ['tree']]:

            if len(organs) < 2:

                self.graph.add_node(organs[0], self.generate_seperate_node(organs))

            else:

                self.graph.add_node('root', self.generate_seperate_node(organs))

        self.graph.add_edge(START, "seed")

        self.graph.add_edge("seed", "root")

        self.graph.add_edge("root", "trunk")

        self.graph.add_edge("trunk", "subtree")

        self.graph.add_edge("subtree", "tree")

    def run(self):

        compiled_graph = self.graph.compile()

        self.final_state = compiled_graph.invoke(self.initial_state)


def dict2str(dict):
    string = "\n".join([f"{key}: {value}" for key, value in dict.items()])
    return string

def extract_op(expression):
    op_list=[]
    # 初始化一个列表，用于存储左括号、右括号和逗号的位置
    positions = []
    # 初始化括号计数器
    bracket_count = 0
    # 遍历表达式字符串
    for i, char in enumerate(expression):
        if char == '(':
            # 如果是左括号，计数器加1
            bracket_count += 1
            if bracket_count == 1:
                # 记录第一个左括号的位置
                positions.append(i)
        elif char == ')':
            # 如果是右括号，计数器减1
            bracket_count -= 1
            if bracket_count == 0:
                # 如果括号计数器为0，说明找到了匹配的右括号，记录位置
                positions.append(i)
        elif char == ',' and bracket_count == 1:
            # 如果是逗号且括号计数器为1，记录逗号位置
            positions.append(i)

    # 如果没有记录任何位置，说明这段表达式中没有算子
    if not positions:
        return op_list

    # 提取第一个左括号之前的所有字符，即算子名称
    op_name = expression[:positions[0]]
    # 将算子名称添加到结果列表中
    op_list.append(op_name)

    # 如果有多个位置记录，说明有嵌套的算子
    if len(positions) > 1:
        # 遍历所有记录的位置，提取子表达式并递归调用
        for start, end in zip(positions[:-1], positions[1:]):
            # 提取子表达式
            sub_expression = expression[start + 1:end]
            # 递归调用函数，传入空列表
            sub_list = extract_op(sub_expression)
            # 将递归调用的结果扩展到主列表中
            op_list.extend(sub_list)
    used_op_info={}
    for op in op_list:
        used_op_info[op]=op_info[op]['description']
    return used_op_info

