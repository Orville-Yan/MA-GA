from openai import OpenAI
from PaperExamples import *
import json

client = OpenAI(api_key="", base_url="https://api.deepseek.com")

class Builder:
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
            user_prompt_ = user_prompt.format(str(article))
            self.user_prompts.append({"role": "user", "content": user_prompt_})
        self.final_prompt = [system_prompt] + self.user_prompts

    def retrieve_response(self):
        response = client.chat.completions.create(
        model="deepseek-chat",
        messages=self.final_prompt,
        stream=False
        )
        return response.choices[0].message.content
    
    def extract_json(self, response:str):
        response_dict = json.loads(response)
        return response_dict

builder = Builder()
response = builder.retrieve_response()
response = builder.retrieve_response(response)
print(response)
