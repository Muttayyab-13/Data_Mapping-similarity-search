import json
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def encode_items(items):
    return model.encode(items)

def save_embeddings(path, embeddings):
    np.save(path, embeddings)

def load_embeddings(path):
    return np.load(path)
