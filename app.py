from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from catalog_service import upload_catalog_to_supabase, check_product_similarity
import logging
from fastapi import Body


# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for the input format
class LineItem(BaseModel):
    description: str
    amount: float
    quantity: int
    price: float

class SimilarityInput(BaseModel):
    invoiceLineItems: List[LineItem]
    poLineItems: Optional[List[LineItem]] = None

class ProductInput(BaseModel):
    product_name: str
    product_description: str

@app.post("/check-similarity")
async def check_similarity(input_data: SimilarityInput):
    """
    Check similarity for each invoice line item, using PO line items as clues if available.
    Returns a list of matches (or no-match results) for each invoice item.
    """
    results = []

    # Iterate over each invoice line item
    for idx, invoice_item in enumerate(input_data.invoiceLineItems):
        # Get the corresponding PO line item description (if available)
        po_description = None
        if input_data.poLineItems and idx < len(input_data.poLineItems):
            po_description = input_data.poLineItems[idx].description

        # Find the closest match for this invoice item
        result = await check_product_similarity(invoice_item, po_description)
        results.append(result)

    return {"matches": results}
# Health check endpoint for Render
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
# Pydantic model for similarity check request and catalog items
class ProductInput(BaseModel):
    product_name: str
    product_description: str


# Uploading JSON FILE (multipart/form-data)
@app.post("/upload-catalog-file")
async def upload_catalog_file(file: UploadFile = File(...)):
    """
    Upload a catalog from a JSON file.
    """
    if not file:
        raise HTTPException(status_code=400, detail="JSON file must be provided")
    
    # You can now handle the uploaded file here
    return await upload_catalog_to_supabase(file, catalog_data=None)

# Uploading RAW JSON (application/json)
@app.post("/upload-catalog-data")
async def upload_catalog_data(catalog_data: List[ProductInput] = Body(...)):
    """
    Upload a catalog from a JSON payload directly.
    """
    if not catalog_data:
        raise HTTPException(status_code=400, detail="Catalog data must be provided")
    
    # Convert list of Pydantic models to list of dicts
    catalog = [item.dict() for item in catalog_data]
    
    return await upload_catalog_to_supabase(file=None, catalog_data=catalog)



if __name__ == "__main__":
    import uvicorn
    print("🚀 Running OCR API on http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=7000)
