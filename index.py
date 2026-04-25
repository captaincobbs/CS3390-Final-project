# Initializes the database
import json
import chromadb
from sentence_transformers import SentenceTransformer


def load_data():
    with open("data/computer_science.json", "r") as file:
        data = json.load(file)
    
    documents = []
    for item in data:
        text = f"Question: {item['question']}\nAnswer: {item['answer']}"
        documents.append(text)
    
    return documents

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = load_data()

embeddings = model.encode(documents)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("cs_rag")

collection.add(
    documents=documents,
    embeddings=embeddings.tolist(),
    ids=[str(i) for i in range(len(documents))]
)

print(f"Indexed {len(documents)} documents")