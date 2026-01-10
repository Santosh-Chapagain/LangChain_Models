from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
)


model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    template="Write detailed report on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Write 50 word summary on {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result= chain.invoke({"topic": "Black Hole"})
print(result)