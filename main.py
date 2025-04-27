import json
import numpy as np
from config import supabase, THRESHOLD
from utils import embed_product, cosine_distance

# === NEW PRODUCT DATA ===
new_product_name = "Nike air"
new_product_description = "Sneakers with a sleek design."

# Step 1: Create embedding
new_embedding = embed_product(new_product_name, new_product_description)

# Step 2: Query existing catalog
result = supabase.table("product_catalog").select("id, product_name, product_description, embedding").execute()

# Step 3: Find closest match
best_match = None
best_dist = float('inf')

for row in result.data:
    # Deserialize the embedding if it's a string (convert it to a list of floats)
    catalog_vector = np.array(json.loads(row["embedding"]), dtype=np.float32)  # Convert string to list of floats
    
    dist = cosine_distance(new_embedding, catalog_vector)
    
    if dist < best_dist:
        best_dist = dist
        best_match = row

# Step 4: Decision based on threshold
if best_dist <= THRESHOLD:
    print(f"✅ Found similar product:")
    print(f"    Name: {best_match['product_name']}")
    print(f"    Description: {best_match['product_description']}")
    print(f"    Distance: {best_dist:.4f}")
else:
    print(f"🚫 No good match found. Adding '{new_product_name}' to catalog.")
    
    # Insert into catalog
    supabase.table("product_catalog").insert({
        "product_name": new_product_name,
        "product_description": new_product_description,
        "embedding": new_embedding  # Insert directly if it's already a list
    }).execute()

    print(f"✅ Inserted '{new_product_name}' successfully!")