import requests


def detect_intent(query):

    prompt = f"""
You are an intent classifier for an e-commerce chatbot.

Classify the user query into exactly ONE category.

Categories:

FAQ
RECOMMENDATION
PRODUCT
REVIEW

FAQ:
Questions about order tracking, delivery, cancellation,
returns, payments or customer support.

RECOMMENDATION:
The user wants suggestions or product recommendations.

PRODUCT:
The user wants information about a specific product,
such as price, brand, category, description or rating.

REVIEW:
The user asks about customer reviews, opinions,
ratings or sentiment.

User Query:
{query}

Return ONLY one category:
FAQ
RECOMMENDATION
PRODUCT
REVIEW
"""


    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )


    intent = response.json()["response"].strip().upper()


    if "RECOMMENDATION" in intent:
        return "RECOMMENDATION"

    elif "REVIEW" in intent:
        return "REVIEW"

    elif "FAQ" in intent:
        return "FAQ"

    elif "PRODUCT" in intent:
        return "PRODUCT"

    return "PRODUCT"
if __name__ == "__main__":

    while True:

        query = input("\nQuestion: ")

        if query.lower() == "exit":
            break

        print("Detected Intent:", detect_intent(query))