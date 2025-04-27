import json
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from config import supabase, THRESHOLD
from utils import embed_product, cosine_distance
import tempfile
import os

app = FastAPI()

# Pydantic model for similarity check request
class ProductInput(BaseModel):
    product_name: str
    product_description: str

# API 1: Upload catalog
@app.post("/upload-catalog")
async def upload_catalog(file: UploadFile = File(...)):
    """
    Upload a catalog.json file and insert products into Supabase.
    """
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="File must be a JSON file")

    # Save uploaded file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Load catalog
        try:
            with open(temp_file_path, 'r') as f:
                catalog = json.load(f)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format")

        # Process each product
        for product in catalog:
            product_name = product.get("product_name")
            product_description = product.get("product_description")
            
            if not product_name or not product_description:
                continue
            
            # Generate embedding
            embedding = embed_product(product_name, product_description)
            
            # Insert into Supabase
            try:
                supabase.table("product_catalog").insert({
                    "product_name": product_name,
                    "product_description": product_description,
                    "embedding": embedding
                }).execute()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to insert '{product_name}': {str(e)}")

        # Clean up temporary file
        os.unlink(temp_file_path)
        return {"message": "Catalog uploaded successfully"}

    except Exception as e:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Catalog upload failed: {str(e)}")

# API 2: Check similarity
@app.post("/check-similarity")
async def check_similarity(product: ProductInput):
    """
    Check if a product is similar to existing ones in the catalog.
    """
    # Create embedding
    new_embedding = embed_product(product.product_name, product.product_description)

    # Query existing catalog
    try:
        result = supabase.table("product_catalog").select("id, product_name, product_description, embedding").execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query catalog: {str(e)}")

    # Find closest match
    best_match = None
    best_dist = float('inf')

    for row in result.data:
        # Deserialize the embedding
        catalog_vector = np.array(json.loads(row["embedding"]), dtype=np.float32)
        
        dist = cosine_distance(new_embedding, catalog_vector)
        
        if dist < best_dist:
            best_dist = dist
            best_match = row

    # Decision based on threshold
    if best_dist <= THRESHOLD:
        return {
            "status": "match_found",
            "product_name": best_match["product_name"],
            "product_description": best_match["product_description"],
            "distance": best_dist
        }
    else:
        # Insert into catalog
        try:
            supabase.table("product_catalog").insert({
                "product_name": product.product_name,
                "product_description": product.product_description,
                "embedding": new_embedding
            }).execute()
            return {
                "status": "no_match",
                "message": f"Inserted '{product.product_name}' into catalog"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to insert product: {str(e)}")