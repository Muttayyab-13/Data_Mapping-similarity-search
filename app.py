import os
import numpy as np
from utils import load_json, encode_items, save_embeddings, load_embeddings
import chromadb
from chromadb.config import Settings
import json

# === Setup DB ===
#db_dir = "db/chroma_db"
import chromadb

# ✅ Modern way to initialize Chroma
client = chromadb.PersistentClient(path="db/chroma_db")

# Get or create a collection
collection = client.get_or_create_collection(name="product_catalog")


# === Load and Embed Catalog ===
catalog = load_json("data/catalog.json")
embeddings_path = "embeddings/catalog_vectors.npy"

if not os.path.exists(embeddings_path):
    catalog_vectors = encode_items(catalog)
    save_embeddings(embeddings_path, catalog_vectors)
else:
    catalog_vectors = load_embeddings(embeddings_path)

# Add to DB (if empty)
if len(collection.get()["ids"]) == 0:
    for i, name in enumerate(catalog):
        collection.add(
            ids=[f"prod_{i}"],
            documents=[name],
            embeddings=[catalog_vectors[i].tolist()]
        )

# === Match New Products ===
new_products = load_json("data/new_products.json")
threshold = 0.3  # cosine distance

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

for item in new_products:
    vector = model.encode(item)
    result = collection.query(
        query_embeddings=[vector.tolist()],
        n_results=1
    )
    dist = result['distances'][0][0]
    best_match = result['documents'][0][0]

    if dist <= threshold:
        print(f"✅ Matched: {item} -> {best_match} (dist: {dist:.4f})")
    else:
        print(f"🚫 No match for: {item}. Adding to catalog...")
        new_id = f"prod_{len(collection.get()['ids'])}"
        collection.add(
            ids=[new_id],
            documents=[item],
            embeddings=[vector.tolist()]
        )
        catalog.append(item)
        catalog_vectors = np.vstack([catalog_vectors, vector])

# Save updated catalog
with open("data/catalog.json", "w") as f:
    json.dump(catalog, f, indent=2)
save_embeddings(embeddings_path, catalog_vectors)
print("\nCatalog updated!")
