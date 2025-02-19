import os
# from langchain.llms import OpenAI
from group_behavior_prompt import *
import json
from openai import OpenAI
from pathlib import Path
parent_path = Path(__file__).parent.parent
sys.path.append(r'{}'.format(parent_path))

# os.environ["OPENAI_API_KEY"] = "your-deepseek-api-key"

client = OpenAI(api_key="", base_url="https://api.deepseek.com")

class Behavior_Builder:
    def __init__(self, articles:list):
        self.LLM = client
        self.articles : list = articles
        self.prompt_builder()
        pass

    def prompt_builder(self):
        system_prompt_ = {"role": "system", "content": system_prompt}
        self.user_prompts = []
        for i in range(len(self.articles)):
            article = self.articles[i]
            user_prompt_ = user_prompt.format(x = str(i+1), summary=str(article))
            self.user_prompts.append({"role": "user", "content": user_prompt_})
        self.final_prompt = [system_prompt_] + self.user_prompts
        
    def retrieve_response(self):
        response = client.chat.completions.create(
        model="deepseek-chat",
        messages=self.final_prompt,
        stream=False,
        response_format={
        'type': 'json_object'
        }
        )
        return response.choices[0].message.content
    
    def extract_json(self, response:str):
        response_dict = json.loads(response)
        return response_dict

builder = Behavior_Builder([paper_example])
response = builder.retrieve_response()
response = builder.extract_json(response)