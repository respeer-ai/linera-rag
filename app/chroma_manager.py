import os
import chromadb
import asyncio
from app.config import settings
from app.embeddings import embedder_async
from app.logger import logger
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
        async def validate_embedding(texts):
            embeddings = await embedder_async.embed_documents_async(texts)
                
            # Handle different embedding formats and ensure they're lists of floats
            # Convert tuples to lists
            # Convert top-level tuple to list
            if isinstance(embeddings, tuple):
                embeddings = list(embeddings)
                
            if isinstance(embeddings, list):
                # Convert any tuple elements to lists
                embeddings = [list(e) if isinstance(e, tuple) else e for e in embeddings]
                
                # If we have a list of lists, validate they contain numbers
                if all(isinstance(e, list) and all(isinstance(x, (int, float)) for x in e) for e in embeddings):
                    return embeddings
                
                # If we have a list of non-lists, try to convert them to lists
                if all(not isinstance(e, list) for e in embeddings):
                    try:
                        return [[float(x)] for x in embeddings]
                    except:
                        raise ValueError("Could not convert embeddings to list of floats")
            
            # If we have a single embedding, wrap it in a list
            try:
                float(embeddings)
                return [[float(embeddings)]]
            except:
                # If all else fails, try to parse it as a string representation of a list
                try:
                    import json
                    parsed = json.loads(embeddings)
                    if isinstance(parsed, list):
                        return [parsed]
                    raise ValueError(f"Could not parse embedding: {type(embeddings)}")
                except:
                    raise ValueError(f"Invalid embedding format: expected list of lists of numbers, got {type(embeddings)}")
            return embeddings
        
        # Create a synchronous wrapper for the async function
        def sync_embedding_function(texts):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(validate_embedding(texts))
            finally:
                loop.close()
            return result
        
        # Set function attributes to avoid ChromaDB errors
        sync_embedding_function.__name__ = "validate_embedding"
        sync_embedding_function.name = "validate_embedding"

        return self.chroma_client.get_collection(
            name=collection_name,
            embedding_function=sync_embedding_function
        )
    
    async def query_index(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the current collection with a question asynchronously"""
        collection = self.get_collection()
        
        query_embedding = await embedder_async.embed_query_async(query)
        
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
    
    async def create_staging_collection(self):
        """Create a new staging collection for updates"""
        # Delete existing staging collection if exists
        collections = self.chroma_client.list_collections()
        if any(col.name == self.staging_collection_name for col in collections):
            self.chroma_client.delete_collection(self.staging_collection_name)
            
        return self.chroma_client.create_collection(
            name=self.staging_collection_name,
            embedding_function=embedder_async.embed_documents_async
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