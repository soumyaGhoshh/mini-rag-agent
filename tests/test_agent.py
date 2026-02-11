import pytest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure project root is in path so we can import agent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.core import Agent
from config import settings

class MockResponse:
    def __init__(self, content):
        self.text = content

@pytest.fixture
def mock_gemini_model():
    with patch("agent.core.genai.GenerativeModel") as mock:
        model_instance = MagicMock()
        mock.return_value = model_instance
        yield model_instance

@pytest.fixture
def mock_search_docs():
    with patch("agent.core.search_docs") as mock:
        yield mock

def test_agent_answer_with_context(mock_gemini_model, mock_search_docs):
    """Test that agent answers when relevant context is found."""
    # Setup
    mock_search_docs.return_value = [
        {"title": "Doc 1", "content": "The sky is blue.", "score": 0.8},
        {"title": "Doc 2", "content": "Grass is green.", "score": 0.6}
    ]
    
    mock_gemini_model.generate_content.return_value = MockResponse("The sky is blue because of Rayleigh scattering.")
    
    agent = Agent()
    response = agent.answer("Why is the sky blue?")
    
    assert response["answer"] == "The sky is blue because of Rayleigh scattering."
    assert "Doc 1" in response["documents_used"]
    assert "Doc 2" in response["documents_used"]
    # Decision might vary based on keywords in query, but should be one of the "answered" ones
    assert "answered" in response["agent_decision"]

def test_agent_no_context(mock_gemini_model, mock_search_docs):
    """Test that agent falls back when no docs are returned."""
    mock_search_docs.return_value = []
    
    agent = Agent()
    response = agent.answer("What is the meaning of life?")
    
    assert response["answer"] == "I don't have enough information in my knowledge base."
    assert response["documents_used"] == []
    assert response["agent_decision"] == "no_relevant_context_found"

def test_agent_low_score_context(mock_gemini_model, mock_search_docs):
    """Test that agent falls back when docs have low relevance score."""
    # Setup: Return docs with score below threshold
    threshold = settings.SIMILARITY_THRESHOLD
    mock_search_docs.return_value = [
        {"title": "Irrelevant Doc", "content": "Something else entirely.", "score": threshold - 0.1}
    ]
    
    agent = Agent()
    response = agent.answer("What is the meaning of life?")
    
    assert response["answer"] == "I don't have enough information in my knowledge base."
    assert response["documents_used"] == []
    assert response["agent_decision"] == "no_relevant_context_found"

def test_agent_explanation_decision(mock_gemini_model, mock_search_docs):
    """Test heuristic decision for explanation."""
    mock_search_docs.return_value = [
        {"title": "FastAPI", "content": "FastAPI is a web framework.", "score": 0.9}
    ]
    mock_gemini_model.generate_content.return_value = MockResponse("FastAPI description.")
    
    agent = Agent()
    response = agent.answer("Explain what is FastAPI?")
    
    assert response["agent_decision"] == "answered_with_explanation"

def test_agent_comparison_decision(mock_gemini_model, mock_search_docs):
    """Test heuristic decision for comparison."""
    mock_search_docs.return_value = [
        {"title": "FastAPI", "content": "FastAPI...", "score": 0.9},
        {"title": "Flask", "content": "Flask...", "score": 0.8}
    ]
    mock_gemini_model.generate_content.return_value = MockResponse("Comparison.")
    
    agent = Agent()
    response = agent.answer("Compare FastAPI vs Flask")
    
    assert response["agent_decision"] == "answered_with_comparison"
