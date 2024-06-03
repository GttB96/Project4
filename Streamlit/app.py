# Code resources can be found at https://docs.streamlit.io/
import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec, Index
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from environment variable
api_key = os.getenv('PINECONE_API_KEY')

pc = Pinecone(
    api_key=api_key
)
index_name = 'coreyapp'
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=786,  # Dimension has to match dimensions in Pinecone
        metric='cosine',  # similar to the metric set in Pinecone
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index_info = pc.describe_index(index_name)
host = index_info['host']
index = Index(name=index_name, host=host, api_key=api_key)
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

# Function to embed user input
def embed_ingredients(ingredients):
    ingredients_str = ' '.join(ingredients)
    embedding = model.encode(ingredients_str, convert_to_tensor=True)
    return embedding

# Streamlit UI
st.title("Recipe Finder")
st.write("Enter ingredients to find the best matching recipes:")

# Input ingredients
ingredients_input = st.text_area("Ingredients (comma-separated)", "tomato, basil, garlic")
if st.button("Find Recipes"):
    if ingredients_input:
        ingredients_list = [ingredient.strip() for ingredient in ingredients_input.split(',')]
        user_embedding = embed_ingredients(ingredients_list)
        
        # Querying Pinecone to get embeddings and metadata
        results = index.query(vector=user_embedding.cpu().numpy().tolist(), top_k=5, include_metadata=True)
        # UI Results
        st.write("Here are the best matching recipes:")
        for match in results['matches']:
            recipe = match['metadata']
            st.write(f"### {recipe['name']}")
            st.write(f"[Link to recipe]({recipe['url']})")
            st.write(f"**Ingredients:** {recipe['ingredients']}")
            st.write(f"**Score:** {match['score']}")
            st.write("---")
