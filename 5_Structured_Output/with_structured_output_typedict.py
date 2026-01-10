from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict , Annotated, Optional,  Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

#schema
class review(TypedDict):

    key_theme: Annotated[list[str] , "Write down all the key themes discussed in review in a list"]
    summary: Annotated[str, "Brief summary of review"] 
    sentiment: Annotated[Literal['pos' , 'neg'], "Give sentiment either Positive , Negative or Neutral"]
    pros: Annotated[Optional[list[str]] , "write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]] , "write down all the cons inside a list"]

structured_model = model.with_structured_output(review)



result = structured_model.invoke("""
    The iPhone 17 Pro Max is a solid phone with great specs, but it’s quite large and heavy for my hands. The battery life is good but nothing extraordinary compared to some Android phones I’ve used. The camera is versatile and produces good photos, but I noticed some slight noise in low light shots. Performance-wise, it’s excellent and apps open instantly. I’m still undecided if I’d upgrade from my older iPhone model next year.”

Pros:

Great specs and performance

Versatile camera

Cons:

Large and heavy

Battery life only average

Low light camera noise
""")

print(result)