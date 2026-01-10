from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field
from typing import Annotated , Optional


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm)


class Person(BaseModel):
    name: str = Field(description="Name of person")
    age: int = Field(gt = 18 , description="Age of the person")
    city: str = Field(description = "Name of the city the person belongs to ")

parser = PydanticOutputParser(pydantic_object=Person)

templete = PromptTemplate(
    template='Generate the name , age  and city of the fictional {place} person \n {format_instruction}' , 
    input_variables=['place'] , 
    partial_variables={'format_instruction': parser.get_format_instructions()}, 
)

chain = templete | model | parser
final_result = chain.invoke({'place': 'Nepal'})
print(final_result)
