import json
from config import supabase
from utils import embed_product

# Path to the catalog JSON file
CATALOG_FILE = "catalog.json"

def load_catalog(file_path):
    """Load product catalog from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        raise
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' contains invalid JSON.")
        raise

def upload_product_catalog():
    """Upload products from catalog.json to Supabase."""
    # Load catalog
    catalog = load_catalog(CATALOG_FILE)
    
    # Process each product
    for product in catalog:
        product_name = product.get("product_name")
        product_description = product.get("product_description")
        
        if not product_name or not product_description:
            print(f"Skipping invalid product: {product}")
            continue
        
        print(f"Processing '{product_name}'...")
        
        # Generate embedding
        embedding = embed_product(product_name, product_description)
        
        # Insert into Supabase
        try:
            supabase.table("product_catalog").insert({
                "product_name": product_name,
                "product_description": product_description,
                "embedding": embedding
            }).execute()
            print(f"✅ Successfully inserted '{product_name}'")
        except Exception as e:
            print(f"❌ Failed to insert '{product_name}': {str(e)}")

if __name__ == "__main__":
    try:
        upload_product_catalog()
        print("✅ Catalog upload completed!")
    except Exception as e:
        print(f"❌ Catalog upload failed: {str(e)}")