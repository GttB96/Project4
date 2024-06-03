# Project4 Recipe Chatbot
## Features
### Recipe suggestions based on input ingredients using AI
### High accuracy named entity recognition (NER)

## Dependencies
### BeautifulSoup
### pandas
### spaCy
### Doccano
### Pinecone
### Streamlit

# Installation and Setup

## go to [Pine](https://app.pinecone.io/) create a username and password, create an index with dimensions 786 and metric cosine, you will have to change the index name and server location with yours in the app.py.

## Clone the Repository
## First, clone the repository from GitHub:

### git clone https://github.com/GttB96/Project4/
### in your terminal: go to repo Project4/ - touch .env - add PINECONE_API_KEY=[YOUR API KEY]
### then go to the repo: cd Project4/streamlit

## Install Dependencies
## To install the required dependencies, run:

pip install -r requirements.txt

## Run the Application
## To start the Streamlit application, run:

streamlit run app.py

## Access the Application
## Open your web browser and go to:

http://localhost:8501

