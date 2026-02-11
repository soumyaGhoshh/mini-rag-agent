#  Mini AI Agent

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Google%20Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Gemini" />
  <img src="https://img.shields.io/badge/Sentence--Transformers-FFD242?style=for-the-badge&logo=huggingface&logoColor=black" alt="Sentence Transformers" />
</p>

A robust, developer-friendly AI agent built with **FastAPI** that leverages **Gemini** and **RAG (Retrieval-Augmented Generation)** to provide grounded, context-aware answers.

---

##  Features

- **Semantic Search**: Utilizes `sentence-transformers` to retrieve highly relevant context from a local knowledge base.
- **Agent Reasoning**: Explicit decision-making logic determines whether to explain, compare, or fallback based on retrieved data.
- **FastAPI Core**: High-performance, REST API.
- **Gemini Powered**: Uses Google's Gemini model for high-quality, grounded responses.
- **Zero Hallucination**: Strict prompting ensures the agent only answers from provided context.

---

##  Setup & Installation

### 1. Clone & Navigate
```bash
git clone https://github.com/soumyaGhoshh/mini-rag-agent
cd mini-rag-agent
```

### 2. Environment Configuration
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_api_key_here
LLM_MODEL=your_api_key_here
```
> [!NOTE]
> Ensure you have a valid Gemini API key from Google AI Studio.

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

##  Running the Agent

### 1. Activate Environment
```bash
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 2. Launch Server
```bash
uvicorn main:app --reload
```
The API will be available at `http://localhost:8000`. Interactive docs: `http://localhost:8000/docs`.

---

##  Troubleshooting

### Port Conflict
If port 8000 is occupied, you can kill the existing process:
```bash
lsof -ti:8000 | xargs kill -9
```
Or run on a different port:
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

---