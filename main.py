import os
import dotenv
import streamlit as st
import requests
from PIL import Image
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
import mimetypes

# Load environment variables
dotenv.load_dotenv()

# Azure Search configuration
AZURE_SEARCH_ENDPOINT = "https://visionrag.search.windows.net"
AZURE_SEARCH_IMAGES_INDEX = "images-index"
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY", "aMXKjrW8jOP8KGkENvXzed3p09VIZmP5qZWY3G6zdOAzSeDEHJ5z")

# Azure Computer Vision configuration
AZURE_COMPUTER_VISION_URL = "https://visionrag2.cognitiveservices.azure.com/computervision/retrieval"
AZURE_COMPUTER_VISION_KEY = os.getenv("AZURE_COMPUTER_VISION_KEY", "21f28922a02b4a0d9edfccfe13104553")

# Initialize Azure Search client
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_IMAGES_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

GREETINGS = "Hello! I am the Fedway Assistant. I can help you find product images. Please ask me about any product and I will display the images for you."

# Function to get image embedding from Azure Computer Vision
def get_model_params():
    return {"api-version": "2023-02-01-preview", "modelVersion": "latest"}

def get_auth_headers():
    return {"Ocp-Apim-Subscription-Key": AZURE_COMPUTER_VISION_KEY}

def get_image_embedding(image_file):
    mimetype = mimetypes.guess_type(image_file)[0]
    url = f"{AZURE_COMPUTER_VISION_URL}:vectorizeImage"
    headers = get_auth_headers()
    headers["Content-Type"] = mimetype
    with open(image_file, "rb") as f:
        response = requests.post(url, headers=headers, params=get_model_params(), data=f)
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.json()}")
        return None
    return response.json()["vector"]

def get_text_embedding(text):
    url = f"{AZURE_COMPUTER_VISION_URL}:vectorizeText"
    response = requests.post(url, headers=get_auth_headers(), params=get_model_params(), json={"text": text})
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.json()}")
        return None
    return response.json()["vector"]

# Function to add image vectors to search index
def index_images():
    for image_file in os.listdir("images"):
        image_embedding = get_image_embedding(f"images/{image_file}")
        if image_embedding:
            search_client.upload_documents(documents=[{
                "id": image_file.split(".")[0],
                "filename": image_file,
                "embedding": image_embedding
            }])

# Function to search images based on a query
def search_images(query):
    query_vector = get_text_embedding(query)
    if not query_vector:
        return None

    # Perform vector search
    results = search_client.search(
        search_text=None,
        vector_queries=[VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=3,
            fields="embedding"
        )]
    )
    all_results = [doc["filename"] for doc in results]
    if all_results:
        img_path = f"images/{all_results[0]}"
        img = Image.open(img_path)
        return img
    return None

# Streamlit response generator

def response_generator(prompt):
    image_response = search_images(prompt)
    if image_response:
        yield image_response
    else:
        yield None

def greetings_generator(prompt):
    yield GREETINGS
        
# Main Streamlit function
if __name__ == '__main__':
    st.image("fedway-logo.png", use_column_width=False, width=300)
    st.write_stream(greetings_generator("Greetings"))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Capture user input
    if prompt := st.chat_input("What is up?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Generate assistant's response
        image_response = None
        for response in response_generator(prompt):
            image_response = response
        if image_response:
            with st.chat_message("assistant"):
                st.image(image_response)
                st.session_state.messages.append({"role": "assistant", "content": "Image displayed."})
        else:
            st.session_state.messages.append({"role": "assistant", "content": "No matching images found."})
