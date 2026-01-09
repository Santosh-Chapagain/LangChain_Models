from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
#IN chattemplete we dont use humanmessege , systemmessge module . Insted of it we direct pass the tuple and we can make dynamic .
chat_template = ChatPromptTemplate([
    ('system' , 'You are {domain} expert'),
    ('human' , 'Explain in simple terms , what is {topic}')
])
prompt = chat_template.invoke({'domain': 'Physics' , 'topic': 'Kinetic Energy'})
print(prompt)
