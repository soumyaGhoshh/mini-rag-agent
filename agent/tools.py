from typing import List, Dict, Any
from knowledge.store import KnowledgeBase

# Initialize KnowledgeBase (loaded on startup)
kb = KnowledgeBase()

def search_docs(query: str) -> List[Dict[str, Any]]:
    """
    Search the knowledge base for relevant documents.
    Returns a list of dictionaries containing title and content.
    """
    results = kb.retrieve(query, top_k=3)
    formatted_results = []
    
    for doc in results:
        # Include score for debugging/decision logic if needed, 
        # but the tool output format is primarily title/content for the agent
        formatted_results.append({
            "title": doc['title'],
            "content": doc['content'],
            "score": doc['score']
        })
    
    return formatted_results
