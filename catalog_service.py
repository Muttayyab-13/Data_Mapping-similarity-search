import json
import numpy as np
import tempfile
import os
from fastapi import UploadFile, HTTPException
from config import supabase, THRESHOLD
from utils import embed_product, cosine_distance
from typing import Optional


async def upload_catalog_to_supabase(file: UploadFile = None, catalog_data: list = None):
    """
    Process and upload a catalog to Supabase from either a JSON file or a JSON payload.
    """
    # Load catalog from file or use provided catalog_data
    try:
        if file:
            # Handle file upload
            if not file.filename.endswith(".json"):
                raise HTTPException(status_code=400, detail="File must be a JSON file")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name

            try:
                with open(temp_file_path, 'r') as f:
                    catalog = json.load(f)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON format")
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        elif catalog_data:
            # Validate catalog_data
            for item in catalog_data:
                if not isinstance(item, dict) or "product_name" not in item or "product_description" not in item:
                    raise HTTPException(status_code=400, detail="Invalid catalog data: each item must be a dict with 'product_name' and 'product_description'")
            catalog = catalog_data
        else:
            raise HTTPException(status_code=400, detail="Either a JSON file or catalog data must be provided")

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

        return {"message": "Catalog uploaded successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Catalog upload failed: {str(e)}")
    


async def check_product_similarity(invoice_item, po_description: str = None):
    """
    Check if an invoice item (with optional PO description as a clue) is similar to existing products in the catalog.
    """
    # Combine invoice description with PO description (if provided) for better embedding
    description_to_embed = invoice_item.description
    if po_description:
        description_to_embed += f" {po_description}"

    # Create embedding using the combined description
    new_embedding = embed_product(description_to_embed, "")

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
        
        # Ensure new_embedding is a NumPy array for cosine distance calculation
        new_embedding_array = np.array(new_embedding, dtype=np.float32) if not isinstance(new_embedding, np.ndarray) else new_embedding
        
        dist = cosine_distance(new_embedding_array, catalog_vector)
        
        if dist < best_dist:
            best_dist = dist
            best_match = row

    # Decision based on threshold
    if best_dist <= THRESHOLD:
        return {
            "status": "match_found",
            "invoice_description": invoice_item.description,
            "matched_product_name": best_match["product_name"],
            "matched_product_description": best_match["product_description"],
            "distance": float(best_dist)  # Convert numpy.float32 to Python float
        }
    else:
        # Insert into catalog
        try:
            # Convert embedding to list for JSON serialization
            embedding_to_store = new_embedding.tolist() if isinstance(new_embedding, np.ndarray) else new_embedding
            supabase.table("product_catalog").insert({
                "product_name": invoice_item.description,
                "product_description": description_to_embed,
                "embedding": embedding_to_store
            }).execute()
            return {
                "status": "no_match",
                "invoice_description": invoice_item.description,
                "message": f"Inserted '{invoice_item.description}' into catalog"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to insert product: {str(e)}")