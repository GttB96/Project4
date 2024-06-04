# Resources can be found at https://docs.pinecone.io/home
import os
import json
from pinecone import Pinecone, ServerlessSpec, Index
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from environment variable
api_key = os.getenv('PINECONE_API_KEY')

# Initialize Pinecone
pc = Pinecone(api_key=api_key)
index_name = 'coreyapp'

# Checking for Index in Pinecone
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # Dimension needs to match the dimension in Pinecone
        metric='cosine', 
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Get the host for the index
index_info = pc.describe_index(index_name)
host = index_info['host']
index = Index(name=index_name, host=host, api_key=api_key)

# Load the embedding model
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

# Load the JSON converted file
json_file_path = '../SpaCy_Extracted_Data/PINECONE_TAGGED_MASTERDATA_Class.json'
with open(json_file_path, 'r', encoding='utf-8') as f:
    recipes = json.load(f)

# Embedding function
def embed_recipe(ingredients):
    ingredients_str = ' '.join([ingredient['ingredient'] for ingredient in ingredients])
    embedding = model.encode(ingredients_str)
    return embedding

# Upload the data to Pinecone
for recipe in recipes:
    embedding = embed_recipe(recipe['tagged_ingredients'])
    metadata = {
        "name": recipe["name"],
        "url": recipe["url"],
        "ingredients": ', '.join([ingredient['ingredient'] for ingredient in recipe['tagged_ingredients']])
    }
    index.upsert([(recipe['url'], embedding, metadata)])

print("Data uploaded to Pinecone successfully.")