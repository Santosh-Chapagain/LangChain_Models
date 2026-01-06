from google import genai
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
print("You want to ask question or not (y/n) ?\n")
choice = input().lower()
while choice == 'y' :
    print("ask something", end=" " )
    result = model.invoke(input())
    print(result.content)
    time.sleep(5)
    print("\nYou want to ask question or not (y/n) ?\n")
    choice = input()




