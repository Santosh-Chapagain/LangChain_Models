from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

prompt = PromptTemplate(
    template= "Write small joke on \n {topic}",
    input_variables=['topic']

)
prompt2 = PromptTemplate(
    template= "Write explanation on following joke - \n {text}",
    input_variables=['text']

)

parser = StrOutputParser()

chain = RunnableSequence(prompt, model, parser, prompt2, model, parser)
result = chain.invoke({'topic': 'Newton Third law'})
print(result)