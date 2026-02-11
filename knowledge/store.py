import json
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import os

class KnowledgeBase:
    def __init__(self, documents_path: str = "knowledge/documents.json"):
        self.documents_path = documents_path
        self.output_dim = 384 # Dimension for all-MiniLM-L6-v2
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.model = None
        
        # Load data immediately
        self.load_documents()
        self.initialize_model()

    def load_documents(self):
        if not os.path.exists(self.documents_path):
            print(f"Warning: Document file not found at {self.documents_path}")
            self.documents = []
            return

        try:
            with open(self.documents_path, 'r') as f:
                self.documents = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {self.documents_path}")
            self.documents = []

    def initialize_model(self):
        if not self.documents:
            print("No documents to embed.")
            return
            
        print("Loading embedding model...")
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Generating embeddings...")
            texts = [doc['content'] for doc in self.documents]
            self.embeddings = self.model.encode(texts)
            # Normalize embeddings for cosine similarity
            norm = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.embeddings = self.embeddings / norm
            print(f"Embeddings generated for {len(self.documents)} documents.")
        except Exception as e:
            print(f"Failed to initialize model or generate embeddings: {e}")
            self.embeddings = None

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.model is None or self.embeddings is None:
            return []

        query_embedding = self.model.encode([query])[0]
        # Normalize query
        norm_query = np.linalg.norm(query_embedding)
        if norm_query == 0:
            return []
        query_embedding = query_embedding / norm_query

        # Compute cosine similarity: dot product of normalized vectors
        scores = np.dot(self.embeddings, query_embedding)
        
        # Get top k indices
        top_k_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_k_indices:
            # You can filter by a threshold here if needed, e.g., score > 0.3
            results.append({
                "score": float(scores[idx]),
                **self.documents[idx]
            })
            
        return results

# Singleton instance can be created here or in dependencies