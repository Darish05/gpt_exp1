import pandas as pd
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load products
products = pd.read_csv("data/products.csv")


# Create searchable text
products["search_text"] = (
    products["ProductName"].fillna("") + " " +
    products["Category"].fillna("") + " " +
    products["Brand"].fillna("") + " " +
    products["Description"].fillna("")
)


# Create TF-IDF vectors
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


def recommend_products(query):

    # Convert query into TF-IDF vector
    query_vector = vectorizer.transform([query])

    # Calculate similarity
    similarities = cosine_similarity(
        query_vector,
        product_vectors
    )[0]

    # Get top 3 products
    top_indices = similarities.argsort()[-3:][::-1]

    print("\nRetrieved Products")

    context = ""

    for index in top_indices:

        product = products.iloc[index]

        print(
            product["ProductName"],
            "| Similarity:",
            round(float(similarities[index]), 2)
        )

        context += f"""
Product ID: {product['ProductID']}
Product Name: {product['ProductName']}
Category: {product['Category']}
Brand: {product['Brand']}
Price: ₹{product['Price']}
Rating: {product['Rating']}
Description: {product['Description']}
"""

    prompt = f"""
You are an e-commerce product recommendation assistant.

User request:
{query}

Retrieved products:
{context}

Choose the BEST product from the retrieved products.

Give:
Product Name:
Brand:
Price:
Rating:
Reason:

Use ONLY the retrieved information.
Do not invent specifications.
"""

    return ask_ollama(prompt)

if __name__ == "__main__":

    query = input("Search Product: ")

    result = recommend_products(query)

    print("\nOllama Recommendation:")
    print(result)