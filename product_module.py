import pandas as pd
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


products = pd.read_csv("data/products.csv")


# Combine product information for retrieval
products["search_text"] = (
    products["ProductName"].fillna("") + " " +
    products["Category"].fillna("") + " " +
    products["Brand"].fillna("") + " " +
    products["Description"].fillna("")
)


vectorizer = TfidfVectorizer(
    stop_words="english"
)

product_vectors = vectorizer.fit_transform(
    products["search_text"]
)


def ask_ollama(prompt):

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]


def product_question(query):

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        product_vectors
    )[0]

    best_index = similarities.argmax()

    score = similarities[best_index]

    product = products.iloc[best_index]


    print("\nRetrieved Product")
    print("-------------------------")
    print("Product ID:", product["ProductID"])
    print("Product Name:", product["ProductName"])
    print("Similarity:", round(float(score), 2))


    # Structured RAG context
    context = f"""
Product ID: {product['ProductID']}
Product Name: {product['ProductName']}
Category: {product['Category']}
Brand: {product['Brand']}
Price: ₹{product['Price']}
Rating: {product['Rating']}
Description: {product['Description']}
"""


    prompt = f"""
You are an e-commerce product assistant.

Answer the user's question ONLY using the context below.

PRODUCT CONTEXT:
{context}

USER QUESTION:
{query}

Rules:
1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not present in the context,
   say "The information is not available in the product details."
4. Keep the answer concise.
"""


    return ask_ollama(prompt)


if __name__ == "__main__":

    while True:

        query = input("\nAsk about a product: ")

        if query.lower() == "exit":
            break

        print("\nAssistant:")
        print(product_question(query))
        