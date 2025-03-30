# Table of Contents
- [Table of Contents](#table-of-contents)
  - [Building a Simple Retrieval-Augmented Generation (RAG) System with LangChain, Azure OpenAI, and Elasticsearch](#building-a-simple-retrieval-augmented-generation-rag-system-with-langchain-azure-openai-and-elasticsearch)
  - [Prerequisites](#prerequisites)
  - [Setup and Installation](#setup-and-installation)
    - [Example Usage:](#example-usage)
      - [1. Ingest Documents](#1-ingest-documents)
      - [2. Query and Generate Response](#2-query-and-generate-response)
    - [Testing the Integration with cURL or Postman](#testing-the-integration-with-curl-or-postman)
      - [cURL Command](#curl-command)
      - [Postman](#postman)
  - [RAG Basics](#rag-basics)
  - [Elastic Vector Database Primer](#elastic-vector-database-primer)
  - [Langchain AzureOpenAIEmbeddings](#langchain-azureopenaiembeddings)
  - [Langchain Elasticsearch Library](#langchain-elasticsearch-library)
  - [Difference between `AzureChatOpenAI` and `AzureOpenAI`](#difference-between-azurechatopenai-and-azureopenai)
    - [AzureOpenAI](#azureopenai)
    - [AzureChatOpenAI](#azurechatopenai)
    - [Summary of Differences:](#summary-of-differences)
  - [Difference between the Completions API and Chat Completions API](#difference-between-the-completions-api-and-chat-completions-api)
    - [Completions API Example (Used for single prompts)](#completions-api-example-used-for-single-prompts)
    - [Chat Completions API Example (Used for multi-turn chat)](#chat-completions-api-example-used-for-multi-turn-chat)
    - [Key Takeaways](#key-takeaways)
    - [Why Did You Get the Error?](#why-did-you-get-the-error)


## Building a Simple Retrieval-Augmented Generation (RAG) System with LangChain, Azure OpenAI, and Elasticsearch
This tutorial covered the basics of RAG, integrating LangChain with Azure OpenAI and Elasticsearch, and running a simple Flask API. With these tools, you can build intelligent applications that leverage powerful LLMs like Azure OpenAI while efficiently managing large datasets with Elasticsearch.

## Prerequisites

To get started, ensure you have the following:

- **Python**: Version 3.8 or later.
- **Azure OpenAI API**: Credentials for accessing Azure OpenAI models.
- **Elasticsearch**: Access to an Elasticsearch instance (Elastic Cloud).

## Setup and Installation

1. **Create a Virtual Environment**:

   Create a virtual environment to manage dependencies for your project:

   ```bash
   python -m venv venv
   ```

2. **Activate the Virtual Environment**:

   On Windows:

   ```bash
   .\venv\Scripts\activate
   ```

   On macOS/Linux:

   ```bash
   source venv/bin/activate
   ```
3. **Install Dependencies**:

   Install the necessary Python packages:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:

   Create a `.env` file in the project root and add your credentials:

   ```env
   AZURE_OPENAI_API_KEY="your-api-key"
   AZURE_OPENAI_ENDPOINT="https://<your-endpoint>.openai.azure.com/"
   AZURE_OPENAI_API_VERSION="2024-02-01"
   ELASTIC_CLOUD_ID="your-elastic-cloud-id"
   ELASTICSEARCH_API_KEY="your-elastic-api-key"
   ELASTICSEARCH_INDEX="your-index-name"
   AZURE_DEPLOYMENT_NAME="your-azure-deployment-name"
   ```

5. **Run the flask app**
   If you were not running the Flask application, now it is a good time to start it. Change to the project directory in a terminal window, activate the Python virtual environment, and then start the application with:
   ```bash
   python app.py
   ```

### Example Usage:

#### 1. Ingest Documents
To ingest documents, you can send a POST request to `/ingest` with a document’s `id` and `content`:

```json
{
  "id": "1",
  "content": "How to reset your password on the platform?"
}
```

#### 2. Query and Generate Response
To retrieve and generate a response, send a POST request to `/retrieve` with a query:

```json
{
  "query": "How can I change my password?"
}
```

The system will first retrieve the most relevant documents and then generate a response using the Azure OpenAI model.

### Testing the Integration with cURL or Postman

You can use cURL or Postman to test the integration.

#### cURL Command

Run the following cURL command to test the `/retrieve` endpoint:

```bash
curl --location 'http://127.0.0.1:5000/retrieve' \
--header 'Content-Type: application/json' \
--data '{
  "query": "How can I change my password for Jira?"
}'
```

#### Postman

Alternatively, you can use Postman to make the same request by setting:

- **Method**: `POST`
- **URL**: `http://127.0.0.1:5000/retrieve`
- **Headers**:
  - `Content-Type: application/json`
- **Body** (raw JSON):
  ```json
  {
    "query": "How can I change my password for Jira?"
  }
  ```
---


## RAG Basics

Retrieval-Augmented Generation (RAG) is a machine learning framework where a language model uses external data (often from a database or API) to enhance its responses. It combines two key components:

1. **Retrieval**: Retrieve relevant documents or data based on the user's query.
2. **Generation**: Use a generative model (like GPT) to generate a response based on the retrieved information.

This approach is especially powerful when combined with large-scale embeddings and search systems like Elasticsearch to provide real-time, data-driven responses.

## Elastic Vector Database Primer

Elasticsearch is a distributed search and analytics engine that is commonly used for storing and querying vector embeddings. In the context of RAG, we use Elasticsearch to store and efficiently retrieve document embeddings generated by a model like Azure OpenAI. Elasticsearch allows for:

- **Approximate Nearest Neighbor (ANN) Search**: A fast method for retrieving vectors that are most similar to a given query vector.
- **Exact Search**: For precise, full-text matching.
- **Sparse Vector Search**: To deal with sparse embeddings and improve search performance.
  

## Langchain AzureOpenAIEmbeddings

[AzureOpenAIEmbeddings Reference Doc](https://python.langchain.com/api_reference/openai/embeddings/langchain_openai.embeddings.azure.AzureOpenAIEmbeddings.html#langchain_openai.embeddings.azure.AzureOpenAIEmbeddings)

`AzureOpenAIEmbeddings` is a class within LangChain that facilitates integration with Azure OpenAI's embedding models. Embeddings are numerical representations of text that capture semantic meaning, enabling tasks like semantic search, clustering, and recommendations. This class allows developers to generate such embeddings using models hosted on Azure's OpenAI service.

**Key Features:**

- **Seamless Integration:** `AzureOpenAIEmbeddings` provides a straightforward interface to access Azure-hosted OpenAI embedding models, streamlining the process of generating text embeddings.

- **Customization:** Users can specify parameters such as the model name (e.g., `text-embedding-3-large`), the number of dimensions for the embeddings (if supported by the model), and connection details like the Azure endpoint and API key.

- **Environment Variable Support:** The class can read configuration details from environment variables, simplifying setup and enhancing security by avoiding hard-coded credentials.

**Setup and Usage:**

1. **Install the Required Package:**

   Ensure you have the `langchain-openai` integration package installed:

   ```bash
   pip install langchain-openai
   ```

2. **Configure Environment Variables:**

   Set the necessary environment variables with your Azure OpenAI credentials:

   ```bash
   export AZURE_OPENAI_API_KEY="your-api-key"
   export AZURE_OPENAI_ENDPOINT="https://<your-endpoint>.openai.azure.com/"
   export AZURE_OPENAI_API_VERSION="2024-02-01"
   ```

3. **Instantiate the `AzureOpenAIEmbeddings` Class:**

   ```python
   from langchain_openai import AzureOpenAIEmbeddings

   embeddings = AzureOpenAIEmbeddings(
       model="text-embedding-3-large"
       # Optionally specify other parameters like dimensions, azure_endpoint, and api_key
   )
   ```

4. **Generate Embeddings:**

   - **For a Single Query:**

     ```python
     text = "This is a test document."
     query_embedding = embeddings.embed_query(text)
     ```

   - **For Multiple Documents:**

     ```python
     documents = ["Document 1 text.", "Document 2 text."]
     document_embeddings = embeddings.embed_documents(documents)
     ```

**Important Considerations:**

- **Model Compatibility:** Ensure that the model specified supports embedding operations. For instance, models like `text-embedding-ada-002`, `text-embedding-3-large`, and `text-embedding-3-small` are designed for embeddings. Using an incompatible model, such as `gpt-4`, will result in errors indicating that the embeddings operation is not supported. citeturn0search5

- **Parameter Mutability:** Some parameters, like `azure_endpoint` and `api_key`, can be set directly when instantiating the class or through environment variables. However, avoid specifying both simultaneously to prevent conflicts.

By leveraging `AzureOpenAIEmbeddings`, developers can effectively utilize Azure's powerful embedding models within the LangChain framework, enabling advanced natural language processing capabilities in their applications.

## Langchain Elasticsearch Library

[Elasticsearch Reference Doc](https://python.langchain.com/docs/integrations/vectorstores/elasticsearch/)

[Elasricstore API Reference Doc](https://python.langchain.com/api_reference/elasticsearch/vectorstores/langchain_elasticsearch.vectorstores.ElasticsearchStore.html#langchain_elasticsearch.vectorstores.ElasticsearchStore)

LangChain's integration with Elasticsearch enables the storage and retrieval of vector embeddings within Elasticsearch, facilitating efficient similarity searches and enhancing applications like Retrieval-Augmented Generation (RAG).
**ElasticsearchStore:**
The primary component of this integration is the `ElasticsearchStore` class. It offers various retrieval strategies:

- **Approximate Nearest Neighbor (ANN) Search:** Utilizes the Hierarchical Navigable Small World (HNSW) algorithm for fast and memory-efficient searches.
  
- **Exact Search:** Performs brute-force searches using the `script_score` function for precise results.
  
- **Sparse Vector Search:** Employs text expansion models like ELSER for sparse vector retrieval.

These strategies allow developers to tailor search behavior to their specific application needs. citeturn0search1

**Installation:**
To use this integration, install the `langchain-elasticsearch` package:


```bash
pip install langchain-elasticsearch
```

Additionally, ensure that the `elasticsearch` package is installed to manage connections with your Elasticsearch instance.

**Usage Example:**
Here's how to initialize the `ElasticsearchStore` with OpenAI embeddings:


```python
from langchain_elasticsearch.vectorstores import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings

# Initialize the embedding model
embedding = OpenAIEmbeddings()

# Set up the Elasticsearch store
vectorstore = ElasticsearchStore(
    embedding=embedding,
    index_name="langchain-demo",
    es_url="http://localhost:9200",
    strategy=ElasticsearchStore.ApproxRetrievalStrategy()
)
```


This configuration connects to a local Elasticsearch instance and sets up the `ElasticsearchStore` with the approximate retrieval strategy. citeturn0search2

**Deprecation Notice:**
The earlier `ElasticVectorSearch` class is deprecated in favor of `ElasticsearchStore`, which provides enhanced functionality and performance. citeturn0search9


## Difference between `AzureChatOpenAI` and `AzureOpenAI`

[AzureChatOpenAI](https://python.langchain.com/docs/integrations/chat/azure_chat_openai/)

[AzureOpenAI](https://python.langchain.com/docs/integrations/llms/azure_openai/)

The key difference between `AzureChatOpenAI` and `AzureOpenAI` in LangChain lies in how they handle the interaction with the underlying models and APIs, specifically for conversational use cases:

### AzureOpenAI
   - **Purpose**: This class is used to interact with general OpenAI models (like GPT-3 and GPT-4) through Azure's platform. It is typically used for non-conversational (static) tasks such as embeddings generation, text completions, or other NLP-based operations that do not involve an ongoing conversation context.
   - **Features**:
     - It leverages Azure’s OpenAI API to provide access to models like `text-davinci-003`, `text-embedding-ada-002`, etc.
     - It can be used for generating embeddings, text completions, summarizations, or other tasks requiring the power of GPT models but without the specific conversational context.

### AzureChatOpenAI
   - **Purpose**: This class is specifically designed to interact with models optimized for conversational use cases, such as `gpt-3.5-turbo` or `gpt-4` models in Azure, where maintaining a conversation context (memory of previous exchanges) is important.
   - **Features**:
     - It is ideal for building chatbots or conversational agents where the model needs to maintain context over multiple interactions with a user.
     - In this case, the model is expected to understand the flow of conversation, keeping track of what has been said and responding appropriately as part of a back-and-forth dialogue.
     - It is often used for chat-based applications that require conversational AI behavior.

### Summary of Differences:
- **AzureOpenAI**: More general-purpose, used for embeddings and non-conversational tasks.
- **AzureChatOpenAI**: Tailored for building conversational AI that requires context and memory across interactions.

In essence, if your application requires the model to remember previous interactions and maintain a conversational context, you would choose `AzureChatOpenAI`. If you just need a model for generating text or embeddings without conversational context, `AzureOpenAI` would be the better option.

## Difference between the Completions API and Chat Completions API
Sure! Let's look at the **difference between the Completions API and Chat Completions API** using practical examples.  

---

### Completions API Example (Used for single prompts)
✅ **Works with models like `text-davinci-003`**  
❌ **Does NOT work with `gpt-3.5-turbo` or `gpt-4`**  

Example request using Python:  

```python
import openai

response = openai.Completion.create(
    model="text-davinci-003",  # Works only with models that support completions
    prompt="Write a short story about a detective solving a mystery.",
    max_tokens=100
)

print(response["choices"][0]["text"])
```

**Expected Output:**
```
Detective James entered the dimly lit alley, following the faint traces of footprints. 
His keen eyes spotted a dropped handkerchief—embroidered with initials that linked 
to the missing heiress. With a knowing smirk, he realized he was on the right path...
```

✅ The model treats this as a single prompt and generates text in response.  
❌ It does NOT remember any previous conversations or context.  

---

### Chat Completions API Example (Used for multi-turn chat)
✅ **Works with models like `gpt-3.5-turbo` or `gpt-4`**  
❌ **Does NOT work with `text-davinci-003`**  

Example request using Python:  

```python
import openai

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",  # This model only works with chat completion
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the FIFA World Cup in 2018?"},
    ]
)

print(response["choices"][0]["message"]["content"])
```

**Expected Output:**
```
France won the FIFA World Cup in 2018, defeating Croatia 4-2 in the final.
```

✅ The model understands structured chat messages (`system`, `user`, `assistant`).  
✅ It remembers past messages and can maintain conversation context.  
❌ Cannot be used with `text-davinci-003` because it is not optimized for chat-style responses.  

---

### Key Takeaways
| Feature                | Completions API (`text-davinci-003`) | Chat Completions API (`gpt-3.5-turbo`, `gpt-4`) |
|------------------------|-------------------------------------|---------------------------------------------|
| **Input format**       | Single prompt as a string         | List of messages with roles (system, user, assistant) |
| **Multi-turn memory**  | ❌ No memory                        | ✅ Maintains conversation context |
| **Best for**           | Text generation (stories, summaries, Q&A) | Chatbots, AI assistants, multi-turn interactions |
| **Example models**     | `text-davinci-003`                 | `gpt-3.5-turbo`, `gpt-4` |
| **Call method**        | `openai.Completion.create()`       | `openai.ChatCompletion.create()` |

---

### Why Did You Get the Error?
Your error message:  
```
BadRequestError: The completion operation does not work with the specified model, gpt-35-turbo.
```
🔴 **Reason:** You are trying to use `openai.Completion.create()` with `gpt-3.5-turbo`, which **only supports Chat Completions**.  
✅ **Fix:** Use `openai.ChatCompletion.create()` instead.  

Let me know if you need more clarification! 🚀