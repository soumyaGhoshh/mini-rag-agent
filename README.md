# Mini AI Agent

A simple, robust AI agent built with FastAPI that answers questions using a Retrieval-Augmented Generation (RAG) approach. It retrieves relevant information from a local knowledge base and uses an LLM to generate grounded responses.

## Features

- **RAG Architecture**: Retrieves relevant documents using semantic search (Sentence-Transformers).
- **Agent Decision Logic**: Explicitly decides whether to answer, use a tool, or fallback if information is missing.
- **FastAPI Service**: Exposes a clean REST API for querying the agent.
- **Configurable**: Supports OpenAI-compatible LLMs (OpenAI, Azure, Local/Ollama).

## Setup

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment**:
    Create a `.env` file in the root directory (or rename `.env.example` if provided):
    ```env
    OPENAI_API_KEY=your_api_key_here
    # Optional: For local LLMs
    # OPENAI_BASE_URL=http://localhost:11434/v1
    # LLM_MODEL=llama3
    ```

## Running the Agent

### 1. Activating the Virtual Environment
Before running the agent, ensure you are using the provided virtual environment:

```bash
# On macOS/Linux
source venv/bin/activate

# On Windows (Command Prompt)
venv\Scripts\activate
```

### 2. Start the FastAPI server
Once the environment is active, run the following command:

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.
Interactive documentation (Swagger UI) is available at `http://localhost:8000/docs`.

---

## Troubleshooting

### "Address already in use" Error
If you see an error like `[Errno 48] Address already in use`, it means another process is already using port 8000. 

You can kill the existing process using:
```bash
# On macOS/Linux
lsof -ti:8000 | xargs kill -9
```
Or simply run uvicorn on a different port:
```bash
uvicorn main:app --reload --port 8001
```

## Usage

**Endpoint**: `POST /agent/query`

**Request**:
```json
{
  "query": "What is FastAPI?"
}
```

**Response**:
```json
{
  "answer": "FastAPI is a modern, fast (high-performance), web framework for building APIs...",
  "documents_used": ["FastAPI Basics"],
  "agent_decision": "answered_with_explanation"
}
```

## Implementation Logic

### 1. Knowledge Base (RAG)
- **Documents**: Stored in `knowledge/documents.json`.
- **Embeddings**: Generated on startup using `sentence-transformers/all-MiniLM-L6-v2`.
- **Retrieval**: Uses cosine similarity to find the top-k most relevant documents for a given query.

### 2. Agent Workflow
1.  **Receive Query**: The agent receives a user question.
2.  **Tool Usage**: Calls the `search_docs` tool to retrieve potential context.
3.  **Decision Making**:
    -   **Relevance Check**: If the similarity score of the top result is below a threshold (default 0.3), the agent decides it lacks information and returns a fallback message.
    -   **Classification**: If context is found, the agent classifies the intent (Explanation vs Comparison) based on keywords (heuristic for this mini-agent, can be LLM-based).
4.  **Generation**: Constructs a prompt with the retrieved context and queries the LLM to generate the final answer.

## Trade-offs & Assumptions

-   **In-Memory Vector Store**: For simplicity and speed in this "mini" scale, embeddings are stored in memory (NumPy arrays). For production with millions of docs, a dedicated Vector DB like Chroma or Pinecone is required.
-   **Model Loading**: The embedding model is loaded on startup. This increases startup time but ensures fast query processing.
-   **Heuristic Decision**: The decision to "explain" or "compare" is currently based on simple keyword matching for visibility. A more complex agent would use the LLM to classify intent explicitly.
-   **Synchronous Processing**: The current implementation is largely synchronous. For high traffic, async database/vector store calls would be preferred.
