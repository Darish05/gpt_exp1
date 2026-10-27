import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer


# Load FAQ dataset
faqs = pd.read_csv("data/faqs.csv")

print("FAQ dataset loaded.")
print("Number of FAQs:", len(faqs))


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Generate embeddings for FAQ questions
embeddings = model.encode(
    faqs["Question"].tolist(),
    normalize_embeddings=True
)


# Save embeddings
with open("models/faq_embeddings.pkl", "wb") as file:
    pickle.dump(embeddings, file)


print("FAQ embeddings created successfully.")
print("Embedding shape:", embeddings.shape)