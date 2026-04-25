# Ask a question from a model trained on the database
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = SentenceTransformer("all-MiniLM-L6-v2")

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_llm = AutoModelForSeq2SeqLM.from_pretrained(model_name)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("cs_rag")

def ask(question):
    query_embedding = model.encode([question])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3
    )

    context = "\n\n".join(results["documents"][0])

    prompt = f"""
Use the context below as a knowledge base. Do not repeat questions in the answer. Summarize clearly.
If you do not know the answer to a question based on your sources, respond "I don't know"

Context:
{context}

Question: {question}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model_llm.generate(**inputs, max_new_tokens=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer, results["documents"][0]