from flask import Flask, request, jsonify # Flask for building the web server and handling HTTP requests
from flask_cors import CORS # Flask-CORS to enable Cross-Origin Resource Sharing (CORS) for the API
from elasticsearch import Elasticsearch  #  Elasticsearch client to interact with Elastic Cloud
import os # To interact with environment variables
from dotenv import load_dotenv  # To load environment variables from a .env file
from pprint import pprint
from langchain_openai import AzureOpenAIEmbeddings # To use Azure OpenAI embeddings through LangChain
from langchain_openai import AzureChatOpenAI

load_dotenv()

# Initialize Flask app (this will be our API server)
app = Flask(__name__)
CORS(app)

# Connect to Elastic Cloud using the credentials stored in the .env file
es = Elasticsearch(cloud_id=os.environ['ELASTIC_CLOUD_ID'],
                   api_key=os.environ['ELASTICSEARCH_API_KEY'])
client_info = es.info()
pprint(client_info.body)

# Define the index in Elasticsearch where documents will be stored
INDEX_NAME = os.environ['ELASTICSEARCH_INDEX']

# Initialize Azure OpenAI embeddings using LangChain's AzureOpenAIEmbeddings class
# This allows us to use Azure OpenAI to generate vector embeddings for text
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

llm = AzureChatOpenAI(
    # azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    # api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME")
)

# Define a basic route to check if the server is running.
@app.route("/")
def home():
    return "Customer Support Chatbot API is running."

# Define an endpoint to ingest documents (POST request)
@app.route("/ingest", methods=["POST"])
def ingest_document():
    """
    This route handles ingesting a new document into the Elasticsearch database.
    The document is converted to an embedding vector using Azure OpenAI,
    and both the document and its embedding are stored in Elasticsearch.
    """
    # Get the data from the incoming JSON request
    data = request.json
    doc_id = data.get("id") # The document ID (usually a unique identifier)
    content = data.get("content")  # The actual content of the document (e.g., a paragraph or FAQ)

    # If either the document ID or content is missing, return an error
    if not content or not doc_id:
        return jsonify({"error": "Missing required fields"}), 400

    # Generate the vector embedding for the content using Azure OpenAI
    # The 'embed' function returns a list of embeddings, so we take the first one
    embedding = embeddings.embed_query(content)

    # Prepare the document body to be stored in Elasticsearch
    doc_body = {
        "content": content,  # Store the original content of the document
        "vector": embedding   # Store the vector embedding of the document content
    }

    # Store the document and its vector in Elasticsearch under the specified index
    # Each API call will send the doc to store in the index.
    es.index(index=INDEX_NAME, id=doc_id, body=doc_body)

    return jsonify({"message": "Document ingested successfully"})

@app.route("/retrieve", methods=["POST"])
def retrieve_and_generate():
    """
    This route retrieves relevant documents based on the query,
    and generates a response using the LLM (Azure OpenAI).
    """
    # Get the user query from the request
    query = request.json.get("query")
    if not query:
       return jsonify({"error": "Missing query parameter"}), 400
    
    # Perform the retrieval from Elasticsearch based on the query
    # This is elastic native function for search. I am not using langchain elasticstore class.
    query_embedding = embeddings.embed_query(query)


    response = es.search(
        index=INDEX_NAME,
        body={
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                        "params": {
                            "query_vector": query_embedding
                        }
                    }
                }
            }
        }
    )

    # Retrieve the top 3 most relevant documents
    hits = response['hits']['hits'][:3]
    relevant_docs = [hit['_source']['content'] for hit in hits]

    # Combine the query with the relevant documents to form the context for the LLM
    context = "\n".join(relevant_docs)  # Concatenate the relevant documents into a single string

    # Use the LLM to generate a response, using the context (retrieved documents) and the user query
    prompt = f"Here are some documents relevant to your query: \n{context}\n\nBased on this information, answer the following query: {query}"

    
    # Generate the response using the LLM (Azure OpenAI)
    llm_response = llm.invoke([prompt])
    ai_message = llm_response.content

    # Return the response in JSON format
    return jsonify({"response": ai_message})


if __name__ == "__main__":
    app.run(debug=True)