import os
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

load_dotenv()

# Environment Variables
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not url or not key:
    raise ValueError("Supabase URL and Key must be set in environment variables.")

supabase: Client = create_client(url, key)

# Cosine Distance Threshold
THRESHOLD = 0.1  # Adjust this value based on your needs