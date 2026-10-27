import pandas as pd
import pickle
import requests

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load FAQ dataset
faqs = pd.read_csv("data/faqs.csv")


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Load saved FAQ embeddings
with open("models/faq_embeddings.pkl", "rb") as file:
    faq_embeddings = pickle.load(file)


# Send prompt to Ollama
def ask_ollama(prompt):

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:latest",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        if "response" not in data:
            print("\nOllama Response:", data)
            return "Ollama did not return a response."

        return data["response"]

    except requests.exceptions.ConnectionError:
        return "Cannot connect to Ollama. Make sure Ollama is running."

    except requests.exceptions.Timeout:
        return "Ollama request timed out."

    except Exception as e:
        return f"Ollama error: {e}"


# Search for the most relevant FAQ
def faq_search(query):

    # Convert user question into embedding
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    # Calculate cosine similarity
    similarities = cosine_similarity(
        query_embedding,
        faq_embeddings
    )[0]

    # Find best FAQ
    best_index = similarities.argmax()

    best_score = similarities[best_index]

    best_question = faqs.iloc[best_index]["Question"]
    best_answer = faqs.iloc[best_index]["Answer"]

    return best_question, best_answer, best_score


# FAQ module
def faq_module(query):

    question, answer, score = faq_search(query)

    print("\nBest Matching FAQ:", question)
    print("Similarity Score:", round(float(score), 2))

    # Similarity threshold
    threshold = 0.55

    if score < threshold:
        return None

    # Prompt for Ollama
    prompt = f"""
You are an e-commerce customer support assistant.

Answer the user's question using ONLY the FAQ information provided below.

FAQ Question:
{question}

FAQ Answer:
{answer}

User Question:
{query}

Give a short, natural and helpful answer.
Do not invent information.
"""

    return ask_ollama(prompt)


# Main program
if __name__ == "__main__":

    query = input("Ask Question: ")

    result = faq_module(query)

    if result:
        print("\nChatbot:")
        print(result)

    else:
        print("\nNo relevant FAQ found.")