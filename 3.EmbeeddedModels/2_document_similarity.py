from langchain_huggingface import HuggingFaceEmbeddings 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()
embedding = HuggingFaceEmbeddings(
    model='sentence-transformers/all-MiniLM-L6-v2'
)
documents = [
    'Machine learning is a field of computer science that focuses on teaching computers to learn from data without being explicitly programmed. It is widely used in many applications such as speech recognition, image processing, and natural language processing.',
    'Deep learning is a subset of machine learning that uses neural networks with many layers. It has achieved state-of-the-art results in various areas including computer vision, speech recognition, and language translation.',
   'Cooking is the practice of preparing food by combining, mixing, and heating ingredients.There are many different styles of cooking around the world, such as baking, grilling,steaming, and frying',
   'Artificial intelligence is a branch of computer science that aims to create machines capable of intelligent behavior. It includes areas like machine learning, robotics, and expert systems.'
]

query = 'Tell me about machine learning .'
documents_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding] , documents_embedding)[0]
index , score = sorted(list(enumerate(scores)), key = lambda x:x[1])[-1]
print(query)
print(documents[index])
print('Similarity score is : ', score)

