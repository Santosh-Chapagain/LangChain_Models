from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel , Field 
from typing import TypedDict, Annotated, Optional,  Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# schema


class review(BaseModel):

    key_themes: list[str] = Field(
        description="Write down all the key themes discussed in review in a list")

    summary: str = Field(description="Brief summary of review")
    sentiment: Literal["pos", "neg"] = Field(
        description="Give sentiment either Positive , Negative")
    
    pros: Optional[list[str]] = Field(default=None , 
        description="write down all the pros inside a list")
                    
    cons: Optional[list[str]] = Field(default=None , description="write down all the cons inside a list")
    
    name: Optional[str] = Field(default=None , description="Write the name of reviewr")


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
