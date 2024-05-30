import os
import pandas as pd
from pinecone import Pinecone
#from langchain_community.vectorstores 
#import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import(
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder)

import openai
from dotenv import load_dotenv
load_dotenv()


import streamlit as st
import time

from streamlit_chat import message




# read in environment variables and keys
PINE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
#openai.api_key = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# set the Model for encoding using input
MODEL ="text-embedding-ada-002"


# connect to Pinecone vector where data is stored
pc = Pinecone(api_key=PINE_API_KEY)
index = pc.Index("recipechat")



#write title on Streamlit page
st.title("Recipe Bot - Project 4 - Team 1")
st.subheader("Chatbot with Langchain, OpenAI, Pinecone, and Streamlit")
st.subheader("     ")
st.subheader("     ")

# check session states - not actively using yet
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Would you like to search for a recipe?"]


if 'requests' not in st.session_state:
    st.session_state['requests'] = []




# tell user what to input
st.text_input("Enter ingredients or search by recipe name", key="input")


# set query variable to user input and let using know you are searching
query = st.session_state.input

'Searching for ...', query




#print(query)

# embed the users input to prepare for the vector db search
embeddings = OpenAIEmbeddings()
query_embed = embeddings.embed_documents([query])

#print(query_embed)


# query the vector with user input and retrieve the top 5 results
results = index.query(
    vector=query_embed,
    top_k=5,
    include_values=False,
    include_metadata=True

)


# format the results 
text_values = []
for match in results.get('matches', []):
    metadata = match.get('metadata', {})
    text_value = metadata.get('text', None)
    if text_value:
        text_values.append(text_value)

first_two_values_list = []
for text in text_values:
    split_text = text.split(',', 2)
    first_two_values = split_text[:2]
    first_two_values_list.append(first_two_values)

df = pd.DataFrame(first_two_values_list, columns=['Recipe Name', 'URL'])

df_filtered = df[df['URL'].str.startswith('https', na=False)]


# display the results or display no recipes found
if len(df_filtered)==0:
    'Unfortunately there are no recipes matching your search'
else:
        'Results:'
    
        st.dataframe(df_filtered, 
                 hide_index=True,use_container_width=True)


