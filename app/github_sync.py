import os
import re
import asyncio
from git import Repo
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config import settings
from app.embeddings import embedder, embedder_async
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
import hashlib
import shutil

class GitHubSync:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len
        )
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
        self.current_collection_name = "current"
        self.new_collection_name = "new_index"
        
        # Ensure data directories exist
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        os.makedirs(settings.REPOS_DIR, exist_ok=True)
        os.makedirs(settings.CHROMA_DIR, exist_ok=True)
    
    def clone_or_update_repo(self, repo_url: str) -> str:
        """Clone or update a repository and return its local path"""
        repo_name = repo_url.split('/')[-1]
        if repo_name.endswith('.git'):
            repo_name = repo_name[:-4]
        repo_path = os.path.join(settings.REPOS_DIR, repo_name)
        
        if os.path.exists(repo_path):
            repo = Repo(repo_path)
            repo.remotes.origin.pull()
        else:
            Repo.clone_from(repo_url, repo_path)
        
        return repo_path
    
    def process_file(self, file_path: str, repo_name: str) -> List[Dict[str, Any]]:
        """Process a file and return chunks with metadata"""
        if not os.path.isfile(file_path):
            return []
        
        # Skip non-text files
        if not file_path.endswith(('.md', '.txt', '.rs', '.ts')):
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            return []
        
        # Create chunks
        chunks = self.text_splitter.split_text(content)
        
        # Prepare metadata for each chunk
        results = []
        for i, chunk in enumerate(chunks):
            results.append({
                "text": chunk,
                "metadata": {
                    "source": file_path,
                    "repo": repo_name,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })
        
        return results
    
    async def create_index(self, chunks: List[Dict[str, Any]], collection_name: str):
        """Create a new index with the given chunks"""
        if not chunks:
            return
        
        # Create or get collection
        collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.ef
        )
        
        # Prepare documents, metadatas, and IDs
        documents = []
        metadatas = []
        ids = []
        
        # Extract texts for embedding
        texts = [chunk["text"] for chunk in chunks]
        
        # Helper function to safely parse string representations of lists
        def parse_embedding(embedding):
            if isinstance(embedding, str):
                try:
                    # Remove brackets and split by comma
                    return [float(x.strip()) for x in embedding.strip('[]').split(',')]
                except (ValueError, AttributeError):
                    return None
            elif isinstance(embedding, dict) and 'embedding' in embedding:
                return parse_embedding(embedding['embedding'])
            elif isinstance(embedding, list):
                # Handle nested lists by flattening them
                def flatten(lst):
                    result = []
                    for item in lst:
                        if isinstance(item, list):
                            result.extend(flatten(item))
                        else:
                            try:
                                result.append(float(item))
                            except (TypeError, ValueError):
                                continue
                    return result
                flattened = flatten(embedding)
                return flattened if flattened else None
            return None

        # Get embeddings based on configuration
        if embedder_async:
            embeddings = await embedder_async.embed_documents_async(texts)
        else:
            embeddings = embedder.embed_documents(texts)
            
        # Ensure embeddings are properly formatted
        embeddings = [parse_embedding(embedding) for embedding in embeddings]
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk["text"])
            metadatas.append(chunk["metadata"])
            # Create unique ID based on content and metadata
            unique_str = f"{chunk['text']}{str(chunk['metadata'])}"
            ids.append(hashlib.sha256(unique_str.encode()).hexdigest())
        
        # Filter out any None or empty embeddings
        valid_data = [
            (d, m, i, e) for d, m, i, e in zip(documents, metadatas, ids, embeddings)
            if e is not None and len(e) > 0
        ]
        
        # Unzip the valid data
        if valid_data:
            documents, metadatas, ids, embeddings = zip(*valid_data)
            
            # Add to collection
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
        else:
            print("No valid embeddings to add to collection")

    def process_repository(self, repo_url: str) -> List[Dict[str, Any]]:
        """Clone or update a repository and process all its files"""
        repo_path = self.clone_or_update_repo(repo_url)
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        
        # Walk through all files in the repository
        all_chunks = []
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_chunks = self.process_file(file_path, repo_name)
                all_chunks.extend(file_chunks)
        
        return all_chunks
    
    async def update(self):
        """Update all repositories and refresh the index asynchronously"""
        print("Starting repository update...")
        all_chunks = []
        
        for repo_url in settings.REPOSITORIES:
            print(f"Processing repository: {repo_url}")
            chunks = self.process_repository(repo_url)
            all_chunks.extend(chunks)
            print(f"Processed {len(chunks)} chunks from {repo_url}")
        
        # Create new index asynchronously
        print("Creating new index...")
        await self.create_index(all_chunks, self.new_collection_name)
        
        # Atomically swap indexes
        print("Swapping indexes...")
        collections = self.chroma_client.list_collections()
        if self.current_collection_name in [col.name for col in collections]:
            self.chroma_client.delete_collection(self.current_collection_name)
        self.chroma_client.get_collection(self.new_collection_name).modify(name=self.current_collection_name)
        
        print("Index update complete")
    
    def get_query_collection(self):
        """Get the current collection for querying"""
        return self.chroma_client.get_collection(
            name=self.current_collection_name,
            embedding_function=self.ef
        )

# Singleton instance
github_sync = GitHubSync()