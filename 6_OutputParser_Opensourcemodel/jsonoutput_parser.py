from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm = llm)

parser = JsonOutputParser()

templete = PromptTemplate(
    template="Give me the name , age and city of fictional persion \n {format_instruction}" ,
    input_variables=[], #input_variables = variables YOU want to manually provide each time you call .format()
    partial_variables= {'format_instruction': parser.get_format_instructions()} #partial_variables = variables that are always filled automatically with the same value every time — you don’t want to type them every time. 

)
chain = templete | model | parser 
result = chain.invoke({})

print(result)