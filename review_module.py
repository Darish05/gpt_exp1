import pandas as pd
import requests

from transformers import pipeline


# Load reviews
reviews = pd.read_csv("data/reviews.csv")


# Load pre-trained sentiment model
sentiment_model = pipeline(
    "sentiment-analysis"
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


def review_module(product_name):

    # Find reviews for selected product
    product_reviews = reviews[
        reviews["ProductName"].str.lower()
        == product_name.lower()
    ]


    if product_reviews.empty:

        return "No reviews found for this product."


    positive = 0
    negative = 0
    neutral = 0


    review_context = ""


    print("\nDetailed Reviews")
    print("=" * 40)


    for _, row in product_reviews.iterrows():

        review = str(row["Review"])

        result = sentiment_model(review)[0]

        label = result["label"]
        confidence = result["score"]


        if label == "POSITIVE":

            sentiment = "Positive"
            positive += 1

        elif label == "NEGATIVE":

            sentiment = "Negative"
            negative += 1

        else:

            sentiment = "Neutral"
            neutral += 1


        print("-------------------------")
        print("Review:", review)
        print("Sentiment:", sentiment)
        print("Confidence:", round(confidence, 3))


        review_context += f"""
Review: {review}
Rating: {row['Rating']}
Sentiment: {sentiment}
"""


    # Limit context sent to Ollama
    # to avoid an extremely large prompt
    review_context = review_context[:12000]


    prompt = f"""
You are an e-commerce review analysis assistant.

Product:
{product_name}

Customer Reviews:
{review_context}

Generate:

Overall Opinion:
Strengths:
Weaknesses:
Buying Suggestion:

Use only the provided reviews.
Do not invent information.
"""


    summary = ask_ollama(prompt)


    print("\nSentiment Summary")
    print("-------------------------")
    print("Positive:", positive)
    print("Neutral:", neutral)
    print("Negative:", negative)


    print("\nOllama Review Summary")
    print("-------------------------")
    print(summary)

    return summary