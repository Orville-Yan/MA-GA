import os
import sys
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from proxy_prompt import *
from langchain.chains import LLMChain
import json
from pathlib import Path

parent_path = Path(__file__).parent.parent
sys.path.append(str(parent_path))

llm = ChatOpenAI(model_name="deepseek-chat", api_key="", base_url='https://api.deepseek.com')

class Group_behavior_Builder:
    def __init__(self, articles: list):
        self.LLM = llm
        self.articles = articles
        self.prompt_builder()

    def prompt_builder(self):
        self.user_prompts = []
        for i, article in enumerate(self.articles):
            user_prompt_ = PromptTemplate(template=user_prompt, input_variables=['x', 'summary'])
            user_prompt_ = user_prompt_.format(x=str(i + 1), summary=str(article))
            human_message = HumanMessagePromptTemplate.from_template(user_prompt_)
            self.user_prompts.append(human_message)

        system_prompt_ = SystemMessagePromptTemplate.from_template(system_prompt)
        self.final_prompt = ChatPromptTemplate.from_messages([system_prompt_] + self.user_prompts)

    def retrieve_response(self):
        llm_kwargs = {
            'response_format': {
                'type': 'json_object'
            }
        }

        llm_chain = self.final_prompt | self.LLM
        response = llm_chain.invoke({}, llm_kwargs=llm_kwargs)
        return response

    def extract_json(self, response: str):
        response_dict = json.loads(response)
        return response_dict

builder = Group_behavior_Builder([paper_example])


response = builder.retrieve_response()
print(response)
# response_dict = builder.extract_json(response)
# print(response_dict)
