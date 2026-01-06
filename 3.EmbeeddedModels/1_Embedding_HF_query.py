from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    'Kathmandu is the capital of Nepal',
    'Bharatpur is metropolitian city of chitwan',
    'Chitwan lies in bagmati province'
]
vector = embedding.embed_documents(documents)
print(str(vector))
