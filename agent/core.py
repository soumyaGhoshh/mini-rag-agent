from typing import List, Dict, Any
import google.generativeai as genai
from agent.tools import search_docs
from config import settings

class Agent:
    def __init__(self):
        # Initialize Gemini client
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(settings.LLM_MODEL)

    def _construct_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Constructs the prompt for the LLM.
        """
        context_str = "\n\n".join([
            f"Title: {doc['title']}\nContent: {doc['content']}" 
            for doc in context
        ])
        
        prompt = f"""You are a helpful AI assistant. Answer the user's question using ONLY the provided context.
If the context does not contain the answer, say "I don't have enough information in my knowledge base."
Do not make up information or use outside knowledge.

If the user asks for an explanation, explain the concept clearly using the context.
If the user asks for a comparison, compare the concepts using the context from multiple documents.

Context:
{context_str}

Question: 
{query}

Answer:"""
        
        return prompt

    def answer(self, query: str) -> Dict[str, Any]:
        """
        The main agent logic.
        1. Retrieve documents depending on query. 
           (In this simple agent, we always search first to see if we have info).
        2. Check relevance.
        3. Decide to answer or fallback.
        """
        
        # 1. Retrieve (Tool Usage)
        docs = search_docs(query)
        
        # 2. Reasoning / Decision Logic
        # Filter for relevance
        relevant_docs = [
            doc for doc in docs 
            if doc.get('score', 0) >= settings.SIMILARITY_THRESHOLD
        ]
        
        # Explicit Decision: If context is missing
        if not relevant_docs:
            return {
                "answer": "I don't have enough information in my knowledge base.",
                "documents_used": [],
                "agent_decision": "no_relevant_context_found"
            }
            
        # Decision: Answer using context
        # Determine if it's explanation or comparison based on query keywords (simple heuristic)
        # Realistically the LLM handles this via the prompt, but we can log the "intent"
        decision = "answered_using_retrieved_context"
        lower_query = query.lower()
        if "compare" in lower_query or "vs" in lower_query or "difference" in lower_query:
            decision = "answered_with_comparison"
        elif "explain" in lower_query or "what is" in lower_query:
            decision = "answered_with_explanation"

        # 3. Generate Answer
        prompt = self._construct_prompt(query, relevant_docs)
        
        try:
            response = self.model.generate_content(prompt)
            answer_text = response.text
        except Exception as e:
            return {
                "answer": "I encountered an error while generating the response.",
                "documents_used": [],
                "agent_decision": f"llm_error: {str(e)}"
            }

        return {
            "answer": answer_text,
            "documents_used": [doc['title'] for doc in relevant_docs],
            "agent_decision": decision
        }
