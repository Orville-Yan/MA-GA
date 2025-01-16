from LLM.Info import *
from LLM.Direction.DirectionPrompt import *
from langchain.prompts import PromptTemplate
from langchain_community.llms.moonshot import Moonshot
from langgraph.graph import  Graph
from typing import Dict,Any
import copy

organs=['initial','seed','root','branch','trunk','subtree','tree']

initial_state={
    'tree':dict[str:list],
    'op_info':Dict,
    'initial_arg_memorizer':{
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
    'initial_meaning_memorizer':{
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
}

rpn_example=('D_ts_std(D_Minute_weight_mean(M_cs_umr(M_at_div(M_at_div(M_ts_mean_mid_neighbor(M_V, 3), M_ts_mean_left_neighbor(M_V, 3)), M_at_div(M_ts_delay(M_V, 2), M_ts_mean_right_neighbor(M_V, 5))), M_at_add(M_at_div(M_ts_mean_left_neighbor(M_C, 10), M_ts_mean_mid_neighbor(M_L, 2)), M_cs_scale(M_ts_mean_right_neighbor(M_C, 2)))), M_toD_standard(M_cs_rank(M_ts_mean_mid_neighbor(M_L, 2)), D_ts_norm(D_ts_delay(D_L, 20), 2))), 2)')

parser_example={
    'tree':['D_ts_std(ARG0,2)'],
    'subtree':['D_Minute_weight_mean(ARG0,ARG1)'],
    'trunk':['M_cs_umr(ARG0,ARG1)','M_toD_standard(ARG2,ARG3)'],
    'root':[
        'M_cs_umr(M_at_div(M_at_div(ARG0,ARG1), M_at_div(ARG2, ARG3)), M_at_add(M_at_div(ARG4, ARG5, M_cs_scale(ARG6)))','M_toD_standard(M_cs_rank(ARG7), ARG8)'],
    'seed':['M_ts_mean_mid_neighbor(ARG8, 3)',
            'M_ts_mean_left_neighbor(ARG8, 3)',
            'M_ts_delay(ARG8, 2)',
            'M_ts_mean_right_neighbor(ARG8, 5)',
            'M_ts_mean_left_neighbor(ARG5, 10)',
            'M_ts_mean_mid_neighbor(ARG7, 2)',
            'M_ts_mean_right_neighbor(ARG5, 2)',
            'M_ts_mean_mid_neighbor(ARG7, 2)',
            'D_ts_delay(ARG2, 20)',]
}

class Visual:
    def __init__(self,rpn:str):
        self.factor=rpn

    def extract_op(self):

        prompt_template = PromptTemplate(template=OP_extraction_Prompt, input_variables=["op_info","classification","factor_str"])

        prompt = prompt_template.format(op_info=str(op_info), classification=str(Interface_Protocol_Group),
                                          variable_info=str(data_info), factor_str=self.factor)

        response = self.LLM.invoke(prompt)

        self.op_info = eval(response)

    def get_initial_state(self):
        initial_state['tree'] = self.tree

        initial_state['op_info'] = self.op_info

        self.initial_state = initial_state

    def generate_seperate_node(self,organ,node_i):

        def node_func(state):

            prompt_template=PromptTemplate(template=self.prompt_template,
                                           input_variables=['op_info','arg_memorizer','meaning_memorizer','specific','actions'])

            prompt=prompt_template.format(op_info=state['op_info'],
                                          arg_memorizer=state[f'{map(organ)}_arg_memorizer'],
                                          meaning_memorizer=state[f'{map(organ)}_arg_memorizer'],
                                          specific=Specific_Prompt[organ],
                                          actions=self.tree[organ][node_i])

            response=eval(self.LLM.invoke(prompt))

            state[f'{organ}_arg_memorizer'][f'ARG{node_i}']=response['arg_memorizer']

            state[f'{organ}_meaning_memorizer'][f'ARG{node_i}']=response['meaning_memorizer']

            return state

        return node_func

    def generate_union_node(self,organ):

        def node_func(state):
            if "merged_state" not in state:
                state["merged_state"] = copy.deepcopy(state)

            else:
                arg_memory=f'{organ}_arg_memorizer'
                arg_key = list(state[arg_memory].keys())[0]
                arg_value = state[arg_memory][arg_key]

                meaning_memory=f'{organ}_meaning_memorizer'
                meaning_key=list(state[meaning_memory].keys())[0]
                meaning_value = state[meaning_memory][meaning_key]

                state['merged_state'][f'{organ}_arg_memorizer'][arg_key]=arg_value
                state['merged_state'][f'{organ}_meaning_memorizer'][meaning_key] = meaning_value

            return state['merged_state']

        return node_func

    def generate_graph(self):

        graph=Graph()

        graph.add_node('initial',lambda state:state)

        for organ in organs:

            graph.add_node(f'{organ}_union', self.generate_union_node(organ))

            for node_i in range(len(self.tree[organ])):

                graph.add_node(f'{organ}_{node_i}',self.generate_seperate_node(organ,node_i))

                graph.add_edge(f'{organ}_{node_i}',f'{organ}_union')

                if organ!='seed':

                    graph.add_edge(f'{map(organ)}_union',f'{organ}_{node_i}')

                else:
                    graph.add_edge('initial',f'{organ}_{node_i}')

        self.graph=graph

    def run(self):
        self.LLM = Moonshot(model="moonshot-v1-8k")

        self.tree = {'': []}

        self.prompt_template = Visual_Prompt

        self.get_initial_state()

        self.extract_op()


def map(organ):

    index=organs.index(organ)

    return organs[index-1]


