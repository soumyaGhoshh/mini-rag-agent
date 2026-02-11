from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from agent.core import Agent
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mini AI Agent",
    description="A simple AI agent for knowledge retrieval and action execution.",
    version="0.1.0"
)

# Initialize Agent
try:
    agent = Agent()
except Exception as e:
    logger.error(f"Failed to initialize agent: {e}")
    agent = None

class QueryRequest(BaseModel):
    query: str = Field(..., example="What is FastAPI and when should I use it?")

class AgentResponse(BaseModel):
    answer: str
    documents_used: List[str]
    agent_decision: str

@app.post("/agent/query", response_model=AgentResponse)
async def query_agent(request: QueryRequest):
    """
    Endpoint to query the AI agent.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized.")

    try:
        logger.info(f"Received query: {request.query}")
        response = agent.answer(request.query)
        logger.info(f"Agent decision: {response.get('agent_decision')}")
        return response
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
