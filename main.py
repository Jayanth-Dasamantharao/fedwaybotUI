import streamlit as st
import time
import boto3
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_aws import BedrockLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
import os
import dotenv
#from azure.identity import AzureDeveloperCliCredential, get_bearer_token_provider
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmKind,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery
from azure.identity import ClientSecretCredential
from azure.search.documents import SearchClient

dotenv.load_dotenv()

AZURE_SEARCH_SERVICE ="visionrag"
AZURE_SEARCH_ENDPOINT = f"https://visionrag.search.windows.net"
AZURE_SEARCH_IMAGES_INDEX = "images-index"
# Fetch credentials from environment variables or secret store
AZURE_CLIENT_ID = "0ebb4c56-4c18-4f79-882e-d6b08a35441b"
AZURE_CLIENT_SECRET = "j9.8Q~kys~H-sXbLUktipfIhXnJCJVmr3qEaNaXn"
AZURE_TENANT_ID = "3ae6c764-479a-4bd4-b9d7-8bc1832c9fdf"

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

key = "aMXKjrW8jOP8KGkENvXzed3p09VIZmP5qZWY3G6zdOAzSeDEHJ5z"

search_client = SearchClient(AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_IMAGES_INDEX, AzureKeyCredential(key))

import mimetypes
import os

import requests
from PIL import Image

AZURE_COMPUTERVISION_SERVICE = "visionrag2"
AZURE_COMPUTER_VISION_URL = f"https://visionrag2.cognitiveservices.azure.com/computervision/retrieval"
AZURE_COMPUTER_VISION_KEY = "21f28922a02b4a0d9edfccfe13104553"

def get_model_params():
    return {"api-version": "2023-02-01-preview", "modelVersion": "latest"}

def get_auth_headers():
    return {"Ocp-Apim-Subscription-Key": AZURE_COMPUTER_VISION_KEY}

def get_image_embedding(image_file):
    mimetype = mimetypes.guess_type(image_file)[0]
    url = f"{AZURE_COMPUTER_VISION_URL}:vectorizeImage"
    headers = get_auth_headers()
    headers["Content-Type"] = mimetype
    response = requests.post(url, headers=headers, params=get_model_params(), data=open(image_file, "rb"))
    if response.status_code != 200:
        print(image_file, response.status_code, response.json())
    return response.json()["vector"]

def get_text_embedding(text):
    url = f"{AZURE_COMPUTER_VISION_URL}:vectorizeText"
    return requests.post(url, headers=get_auth_headers(), params=get_model_params(),
                         json={"text": text}).json()["vector"]


# ### Add image vectors to search index

for image_file in os.listdir(f"images"):
    image_embedding = get_image_embedding(f"images/{image_file}")
    search_client.upload_documents(documents=[{
        "id": image_file.split(".")[0],
        "filename": image_file,
        "embedding": image_embedding}])
    
# ### Query using an image

from PIL import Image
import time

def search_images(query):
    try:
        query_vector = get_text_embedding(query)
        
        # Perform vector search
        r = search_client.search(
            None,
            vector_queries=[VectorizedQuery(
                vector=query_vector, 
                k_nearest_neighbors=3, 
                fields="embedding"
            )]
        )
        
        all_results = [doc["filename"] for doc in r]
        
        if all_results:
            img_path = "images/" + all_results[0]
            img = Image.open(img_path)  
            return img
        
        else:
            return None  # No matching images found
    
    except Exception as e:
        return None  # Return None in case of any error


def response_generator(prompt):
    try:
        image_response = search_images(prompt)
        if image_response:
            yield image_response  # Yield the image object if found
        else:
            yield None  # Yield None if no image is found or an error occurred
    except Exception as e:
        yield None  # Yield None in case of an exception


# Main streamlit function
import streamlit as st

# Main Streamlit app function
if __name__ == '__main__':
    # Display Fedway logo using Streamlit's st.image with resizing
    st.image("fedway-logo.png", use_column_width=False, width=300)  # Adjust width as needed

    # Embed CSS for additional styling
    st.markdown(
        """
        <style>
            .stChatMessage {
                font-size: 14px;
                padding: 10px;
                border-radius: 8px;
                margin: 5px 0;
            }
            .custom-title {
                font-size: 24px;
                text-align: center;
                font-weight: bold;
                color: #333333;
                margin-top: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display the title with reduced size using custom CSS class
    #st.markdown('<h1 class="custom-title">Fedway Assistant</h1>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Capture user input from the chat input box
    if prompt := st.chat_input("What is up?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate the assistant's response
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(prompt))

        st.session_state.messages.append({"role": "assistant", "content": response})
