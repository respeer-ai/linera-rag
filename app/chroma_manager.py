import os
import chromadb
from app.config import settings
from app.embeddings import embedder
from typing import List, Dict, Any

class ChromaManager:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
        self.current_collection_name = "current"
        self.staging_collection_name = "staging"
        
    def get_collection(self, name: str = None):
        """Get the current production collection"""
        collection_name = name or self.current_collection_name
        # Ensure the embedding function returns a list of floats
        def validate_embedding(texts):
            embeddings = embedder.embed_documents(texts)
            if not all(isinstance(e, list) and all(isinstance(x, (int, float)) for x in e) for e in embeddings):
                raise ValueError("Invalid embedding format: expected list of lists of numbers")
            return embeddings

        return self.chroma_client.get_collection(
            name=collection_name,
            embedding_function=validate_embedding
        )
    
    def query_index(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the current collection with a question"""
        collection = self.get_collection()
        query_embedding = embedder.embed_query(query)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas"]
        )
        
        return [
            {
                "document": doc,
                "metadata": meta,
                "score": score
            }
            for doc, meta, score in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]
    
    def create_staging_collection(self):
        """Create a new staging collection for updates"""
        # Delete existing staging collection if exists
        collections = self.chroma_client.list_collections()
        if any(col.name == self.staging_collection_name for col in collections):
            self.chroma_client.delete_collection(self.staging_collection_name)
            
        return self.chroma_client.create_collection(
            name=self.staging_collection_name,
            embedding_function=embedder.embed_documents
        )
    
    def swap_collections(self):
        """Atomically swap staging and production collections"""
        # Get current collections
        current_collection = self.get_collection(self.current_collection_name)
        staging_collection = self.get_collection(self.staging_collection_name)
        
        # Rename current to temp
        temp_name = "temp_" + self.current_collection_name
        current_collection.modify(name=temp_name)
        
        # Rename staging to current
        staging_collection.modify(name=self.current_collection_name)
        
        # Delete old temp collection
        self.chroma_client.delete_collection(temp_name)
        
        return True

# Singleton instance
chroma_manager = ChromaManager()