import streamlit as st
import time
import os
import boto3
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_aws import BedrockLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String

# AWS Bedrock setup (replace with your keys and region)
os.environ['AWS_ACCESS_KEY_ID'] = st.secrets["AWS_ACCESS_KEY_ID"]
os.environ['AWS_SECRET_ACCESS_KEY'] = st.secrets["AWS_SECRET_ACCESS_KEY"]
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name='us-west-2',
)
model_id = "meta.llama3-70b-instruct-v1:0"

model_kwargs = {
    "temperature": 0.01,
    "top_p": 0.9,
    "max_gen_len": 100
}

llm = BedrockLLM(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
    streaming=True
)

# SQLite database setup
sqlite_uri = 'sqlite:///sample_locations2.db'
engine = create_engine(sqlite_uri)
metadata = MetaData()

# Define and create tables
customers = Table(
    'customers', metadata,
    Column('customer_id', Integer, primary_key=True, autoincrement=True),
    Column('shop_name', String, nullable=False),
)

address = Table(
    'address', metadata,
    Column('address_id', Integer, primary_key=True, autoincrement=True),
    Column('shop_name', String, nullable=False),
    Column('location', String, nullable=False),
    Column('city', String, nullable=False),
    Column('zip_code', String, nullable=False)
)

# Drop existing tables and recreate them
metadata.drop_all(engine)
metadata.create_all(engine)

# Function to populate tables with correct data
def populate_tables():
    with engine.begin() as conn:
        conn.execute(customers.insert(), [
            {'shop_name': 'Buy Rite Liquors'},
            {'shop_name': 'Buffalo Wild Wings'},
            {'shop_name': 'Harvest Wine and Spirits'},
            {'shop_name': 'Kingdom Wine & Spirits'}
        ])
        conn.execute(address.insert(), [
            {'shop_name': 'Buy Rite Liquors', 'location': '1353 Stelton Rd', 'city': 'Piscataway', 'zip_code': '08854'},
            {'shop_name': 'Buffalo Wild Wings', 'location': '625 US-1 South', 'city': 'Iselin', 'zip_code': '08830'},
            {'shop_name': 'Harvest Wine and Spirits', 'location': '2370 Woodbridge Ave', 'city': 'Edison', 'zip_code': '08817'},
            {'shop_name': 'Buy Rite Liquors', 'location': '900 Easton Ave', 'city': 'Somerset', 'zip_code': '08873'},
            {'shop_name': 'Kingdom Wine & Spirits', 'location': '561 US-1 A11', 'city': 'Edison', 'zip_code': '08817'}
        ])

# Populate tables
populate_tables()

# Initialize the SQL database object
db = SQLDatabase.from_uri(sqlite_uri)

# Define schema descriptions for the prompt
schema1 = """
CREATE TABLE customers (
    customer_id integer PRIMARY KEY AUTOINCREMENT,
    shop_name String NOT NULL
);
"""

schema2 = """
CREATE TABLE address (
    address_id integer PRIMARY KEY AUTOINCREMENT,
    shop_name String NOT NULL,
    location String NOT NULL,
    city String NOT NULL,
    zip_code String NOT NULL
);
"""

# Function to generate SQL query with enhanced instructions and validation
def generate_sql_prompt(question):
    template = """ 
    
    You are a SQL expert tasked with generating precise and syntactically correct SQL queries based on the provided table schemas. 
    Follow these guidelines strictly:
    
    1. Only generate the SQL query without explanations or additional text.
    2. Ensure the query is syntactically correct and contains the necessary clauses (`SELECT`, `FROM`, `WHERE`, etc.).
    3. Select columns relevant to answering the question, avoid selecting all columns (`*`) unless explicitly needed.
    4. Always use wildcards "LIKE" to match the shop names.
    5. If the question references specific values, ensure those values are properly quoted and matched accurately in the SQL using the LIKE wildcards.
    6. Avoid using ambiguous natural language constructs; focus on standard SQL syntax.
    7. Do not assume column names; use only those explicitly defined in the provided table schemas.

    Address Table schema:
    {schema2}

    Question: {question}
    SQL Query:

    
    """
    prompt = ChatPromptTemplate.from_template(template)
    conversation = LLMChain(llm=llm, prompt=prompt, verbose=True)
    response = conversation.invoke({"question": question, "schema1": schema1, "schema2": schema2})

    # Clean and validate the SQL query
    cleaned_query = clean_sql_query(response['text'])

    # Check if the cleaned query is valid
    if not validate_sql_query(cleaned_query):
        return fallback_sql_query(question)
    
    return cleaned_query

# Function to clean and extract only SQL components
def clean_sql_query(response):
    # Extract lines that contain valid SQL keywords
    sql_lines = [line for line in response.split('\n') if line.strip().startswith(('SELECT', 'FROM', 'WHERE', 'JOIN', 'ON'))]
    query = ' '.join(sql_lines).strip()

    # Ensure the query ends correctly and is formatted properly
    query = query.split(';')[0] + ';'
    return query

# Function to validate the generated SQL query
def validate_sql_query(query):
    # Basic checks to see if the SQL query contains essential components
    required_keywords = ['SELECT', 'FROM']
    return all(keyword in query.upper() for keyword in required_keywords)

# Fallback SQL query generator for basic scenarios
def fallback_sql_query(question):
    # Simple fallback for common questions to prevent errors
    if "buy rite" in question.lower():
        return "SELECT location, city, zip_code FROM address WHERE shop_name = 'Buy Rite Liquors';"
    # Additional fallbacks can be defined based on common questions or errors
    return "SELECT * FROM address;"  # Generic fallback query

# Prompt for converting SQL response to natural language
def generate_english_response(question, query, result):
    if result == "":
        result = db.run("SELECT * FROM ADDRESS")
    final_prompt = PromptTemplate.from_template(
        """Based on the table schema below, question, SQL query, and SQL response, write an English response of the answer. IF YOU ARE NOT ABLE TO FIND ANY RESPONSE MATCHING WITH THE USER QUESTION, SIMPLY RETURN "SORRY, NO INFORMATION AVAILABLE!":
        Question: {question}
        SQL Query: {query}
        SQL Result: {result}
        Answer: """
    )

    final_chain = (
        RunnablePassthrough.assign(
            question=lambda x: x["user_question"],
            query=lambda x: x["query"],
            result=lambda x: x["answer"]
        )
        | final_prompt
        | llm
        | StrOutputParser()
    )

    return final_chain.invoke({"user_question": question, "query": query, "answer": result}).split('\n')[0]

# Function to run the full query chain with error handling
def sql_complete_chain(user_question):
    sql_query = generate_sql_prompt(user_question)
    try:
        sql_response = db.run(sql_query)
        english_response = generate_english_response(user_question, sql_query, sql_response)
        return english_response
    except Exception as e:
        return f"Error processing SQL query: {e}"

# Streamlit response generator for chat interaction
def response_generator(prompt):
    greetings = ["hi", "hello", "hey", "greetings", "what's up"]
    if prompt.lower().strip() in greetings:
        response = "Hello! How can I assist you today? You can ask me about shop locations or specific details."
    else:
        try:
            response = sql_complete_chain(prompt)
        except Exception as e:
            response = f"Error processing your request: {e}"

    # Stream the response word by word
    for word in response.split():
        yield word + " "
        time.sleep(0.07)

# Main Streamlit app function
# Main Streamlit app function
# Main Streamlit app function
if __name__ == '__main__':
    # Add Fedway logo at the top of the page with reduced size
    st.markdown('<img src="fedway-logo.png" class="logo-img">', unsafe_allow_html=True)

    # Optional: Embed CSS for additional styling
    st.markdown(
        """
        <style>
            .logo-img {
                display: block;
                margin-left: auto;
                margin-right: auto;
                width: 100px;  /* Set your desired width here */
                height: auto;  /* Maintain aspect ratio */
        }
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
    st.markdown('<h1 class="custom-title">Fedway Bot</h1>', unsafe_allow_html=True)

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


