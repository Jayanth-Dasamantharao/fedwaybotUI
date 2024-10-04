import os
import dotenv
import streamlit as st
import requests
from PIL import Image
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
import mimetypes
import boto3
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import LLMChain
from langchain_aws import BedrockLLM

# Load environment variables
#dotenv.load_dotenv()

# Azure Search configuration
AZURE_SEARCH_ENDPOINT = "https://visionrag.search.windows.net"
AZURE_SEARCH_IMAGES_INDEX = "images-index-poc-2"
AZURE_SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]
# Azure Computer Vision configuration
AZURE_COMPUTER_VISION_URL = "https://visionrag2.cognitiveservices.azure.com/computervision/retrieval"
AZURE_COMPUTER_VISION_KEY = os.environ["AZURE_COMPUTER_VISION_KEY"]

# Initialize Azure Search client
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_IMAGES_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

GREETINGS = "Hello! I am the Fedway Assistant. I can help you find product images. Please ask me about any product and I will display the images for you."


def load_llm():
    # Create a Bedrock client
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name='us-east-1',
        aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
    )
    model_id = "meta.llama3-70b-instruct-v1:0"


    model_kwargs = { 
    "temperature": 0.01,
    "top_p": 0.9,
    "max_gen_len" : 250
        
    }
    llm = BedrockLLM(
        client=bedrock_runtime,
        model_id=model_id,
        model_kwargs=model_kwargs,
        streaming=True
    )

    return llm

def llm_response(query):
    llm = load_llm()
    prompt =  """ 

        You are a very intelligent AI assitant who is expert in checking the context of the user query. The user query should match any alcohol/wine related questions or keywords.
        If they match return "Yes".
        If they do not match the context return "No relevant images found. Enter a valid query". 

        Make sure your response is only one of the above two words.

        For example:

        question: "wine and tree"
        answer: "Yes"

        question: "whats the capital of New Jersey?"
        answer: "No relevant images found. Enter a valid query"


        Input: {query}
        Output:
        """

    query_with_prompt = PromptTemplate(
        template = prompt,
        input_variables = ["query"]
    )
    
    llmchain = query_with_prompt | llm 

    response = llmchain.invoke( {"query": query }) 

    return response

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
    allow_image_retrieval = llm_response(prompt)
    if 'No relevant' not in allow_image_retrieval:
        image_response = search_images(prompt)
        if image_response:
            yield image_response
        else:
            yield 'No relevant images found. Enter a valid query'
    else:
        yield allow_image_retrieval

def greetings_generator(prompt):
    yield GREETINGS
        
# Main Streamlit function
if __name__ == '__main__':
    st.image("fedway-logo.png", use_column_width=False, width=300)
    st.title("Fedway Sales - Helpdesk POC")
    index_images()
    st.write_stream(greetings_generator("Greetings"))

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "images" not in st.session_state:
        st.session_state.images = []

    # Display previous chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["content"].startswith("Image reference: "):
                image_index = int(message["content"].split()[-1])  
                st.image(st.session_state.images[image_index], caption=f"Image")
            else:
                st.markdown(message["content"])

    # Capture user input
    if prompt := st.chat_input("What is up?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Generate assistant's response
        image_response = None
        for response in response_generator(prompt):
            if isinstance(response, Image.Image): 
                with st.chat_message("assistant"):
                    st.session_state.images.append(response)
                    image_index = len(st.session_state.images) - 1  
                    st.session_state.messages.append({"role": "assistant", "content": f"Image reference: {image_index}"})
                    st.image(response, caption=f"Image", width=200)
            else:  
                with st.chat_message("assistant"):
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

