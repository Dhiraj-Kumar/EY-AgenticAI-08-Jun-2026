from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

model = OpenAI()

docs = [
    "The Eiffel Tower in Paris is a wrought-iron structure known for its iconic design and panoramic city views.",
    "The Great Wall of China is an ancient series of fortifications built to protect against invasions and spans thousands of miles.",
    "The Statue of Liberty in New York symbolizes freedom and democracy, and was a gift from France to the United States."
]

def get_embedding(text: str):
    response = model.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

query = "Which monument is the symbol of freedom"
doc_embeddings = [get_embedding(doc) for doc in docs]
query_embeddings = get_embedding(query)

score = cosine_similarity([query_embeddings], doc_embeddings)[0]

# print(score)

best_index, best_score = max(enumerate(score), key=lambda x: x[1])

print(docs[best_index])
print(f"Similarity score is : {best_score}")