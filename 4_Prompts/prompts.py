from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate ,load_prompt
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

st.header("Research Tool")

paper_input = st.selectbox('Select Research paper name' , ['Attention Is All You Need' , 'BERT: Pre-training of Deep Bidirectional Transformers' , 'GPT-3: Language Models are Few-Shot Learners','Diffusion Models Beat GANS on Image Synthesis'])

style_input = st.selectbox('Select Explanation Style', ['Beginner-Friendly' , 'Technical' , 'Code-Oriented', 'Mathematical'])

length_input = st.selectbox('Select explanation length' , ['Short(1-2 Paragraphs)' , 'Medium(3-5 paragraphs)' , 'Long(Detailed explanation)'])

style_instructions = {
    'Beginner-Friendly': "Use analogies and simple language. Avoid complex math. Explain like teaching to a college freshman.",

    'Technical': "Include technical details, architecture specifics, formulas, and hyperparameters. Assume reader has ML background.",

    'Code-Oriented': "Provide pseudocode or Python-like snippets. Discuss implementation details, APIs, and practical usage.",

    'Mathematical': "Focus on equations, proofs, and mathematical formulations. Use LaTeX format for equations."
}
length_instructions = {
    'Short(1-2 Paragraphs)': "Be concise. Cover only essential points. Maximum 200-300 words.",

    'Medium(3-5 paragraphs)': "Provide balanced coverage with moderate detail. 400-600 words.",

    'Long(Detailed explanation)': "Comprehensive explanation with examples, context, and detailed analysis. 800+ words."
}
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

#tempelate
template = load_prompt(r"C:\Users\Acer\OneDrive\Desktop\LangChainModel\Prompts\templete.json")

if st.button('Summarize'):
    Chain = template | model
    result = Chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input,
        'length_instructions': length_instructions,
        'style_instructions': style_instructions
    })
    st.write(result.content)
