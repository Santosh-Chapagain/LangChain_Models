from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableBranch
import os
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)
model = ChatHuggingFace(llm=llm)


prompt1 = PromptTemplate(
    template='write a detailed report on {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Summarize the following text in small paragraph \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()
report_gen_chain = RunnableSequence(prompt1 , model , parser)
summary_chain = RunnableSequence(prompt2 , model , parser)
branch_chain = RunnableBranch(
    (lambda x: len(x.split()) >=300 , summary_chain),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain , branch_chain)
result = final_chain.invoke({'topic': 'WorldWar'})
print(result)