import os
# from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from PaperExamples import *
from langchain.chains import LLMChain
import json

# os.environ["OPENAI_API_KEY"] = "your-deepseek-api-key"

llm = ChatOpenAI(model_name="deepseek-chat", api_key="your-deepseek-api-key",base_url='https://api.deepseek.com')

class ExampleBuilder:
    def __init__(self, articles:list):
        self.LLM = llm
        self.articles : list = articles
        self.prompt_builder()
        pass

    def prompt_builder(self):
        self.user_prompts = []
        for i in range(len(self.articles)):
            article = self.articles[i]
            user_prompt_ = PromptTemplate(template=user_prompt, input_variables=['x','summary'])
            user_prompt_ = user_prompt_.format(x = str(i+1), summary=str(article))
            # user_prompt_ = {"summary": str(article)}
            self.user_prompts.append(user_prompt_)
        system_prompt_ = SystemMessagePromptTemplate.from_template(system_prompt)

        self.final_prompt = ChatPromptTemplate.from_messages([system_prompt_].extend(self.user_prompts))


    def retrieve_response(self):
        llm_chain = LLMChain(prompt=self.final_prompt, llm=self.LLM)
        response = llm_chain.run()
        return response
    
    def extract_json(self, response:str):
        response_dict = json.loads(response)
        return response_dict

builder = ExampleBuilder()
response = builder.retrieve_response()
response = builder.retrieve_response(response)
print(response)