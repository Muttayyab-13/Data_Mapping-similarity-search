import numpy as np
from config import client

# Function to create combined embedding
def embed_product(name: str, description: str):
    combined_text = f"{name}. {description}"
    response = client.embeddings.create(
        input=combined_text,
        model="text-embedding-ada-002"
    )
    embedding = response.data[0].embedding   # Correct way to extract the embedding
    return embedding

# Function to compute cosine distance
def cosine_distance(vec1, vec2):
    return 1 - (np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))