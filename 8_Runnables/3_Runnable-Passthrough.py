from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableSequence , RunnablePassthrough
import os
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)
model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="Write small joke on \n {topic}",
    input_variables=['topic']

)
prompt2 = PromptTemplate(
    template="Write explanation on following joke - \n {text}",
    input_variables=['text']

)
parser = StrOutputParser()

joke_chain = RunnableSequence(prompt1 , model , parser)
explanation = RunnableSequence(prompt2 , model , parser)
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': explanation
})

final_chain = RunnableSequence(joke_chain , parallel_chain)

result = final_chain.invoke({'topic': 'AI'})

print(result)