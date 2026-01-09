# Inlangchain , we use placeholder because it helps to store user history . Eg: today user ask : Hello , what is my status of refund ? and stop talking . Again , tommorrow he again ask what about my refund then model have to read past data also . In this case messege holder will take crucial role . 

from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_core.messages import HumanMessage , AIMessage , SystemMessage
#chat template
chat_template = ChatPromptTemplate([
    ('system' , 'You are helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human' , '{query}')
])


chat_history = []
#load chat history
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)


#create prompt
prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'where is my refund ?' })
print(prompt)
