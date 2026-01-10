from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

import os

load_dotenv()

# Use a conversational model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
)

model = ChatHuggingFace(llm=llm)

#1st prompt
templete1 = PromptTemplate(
    template="Write a detailed report on {topic}" ,
    input_variables = ['topic']
)

# 2nd prompt
templete2 = PromptTemplate(
    template="Write a summary about 50 word on following text . /n {text}",
    input_variables=['text']
)

prompt1 = templete1.invoke({'topic': 'black hole'})

result1 = model.invoke(prompt1)

prompt2 = templete2.invoke({'text': result1.content})

result2 = model.invoke(prompt2)

print(result2.content)