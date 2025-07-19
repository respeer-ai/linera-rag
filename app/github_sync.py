import os
import re
import asyncio
from git import Repo
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config import settings
from app.embeddings import embedder_async
import chromadb
from chromadb.utils import embedding_functions
from app.logger import logger
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
    
    async def clone_or_update_repo(self, repo_url: str) -> str:
        """Clone or update a repository and return its local path"""
        repo_name = repo_url.split('/')[-1]
        if repo_name.endswith('.git'):
            repo_name = repo_name[:-4]
        repo_path = os.path.join(settings.REPOS_DIR, repo_name)
        
        # Use asyncio.to_thread to run git operations in a separate thread
        if os.path.exists(repo_path):
            repo = Repo(repo_path)
            repo.remotes.origin.pull()
        else:
            Repo.clone_from(repo_url, repo_path)
        
        return repo_path
    
    async def process_file(self, file_path: str, repo_name: str) -> List[Dict[str, Any]]:
        """Process a file and return chunks with metadata"""
        # Use asyncio.to_thread for file I/O operations
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
        logger.info(f"Creating index {collection_name} with {len(chunks)} chunks")
        if not chunks:
            logger.warning("No chunks provided to create index")
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
            # If it's a list of floats, return as is
            if isinstance(embedding, list) and all(isinstance(x, (int, float)) for x in embedding):
                return embedding
            
            # If it's a tuple, convert to list
            if isinstance(embedding, tuple):
                return list(embedding)
            
            # If it's a string, try to parse it
            if isinstance(embedding, str):
                try:
                    # Try JSON parsing
                    import json
                    parsed = json.loads(embedding)
                    if isinstance(parsed, list) and all(isinstance(x, (int, float)) for x in parsed):
                        return parsed
                except json.JSONDecodeError:
                    try:
                        # Fallback to literal_eval
                        import ast
                        parsed = ast.literal_eval(embedding)
                        if isinstance(parsed, list) and all(isinstance(x, (int, float)) for x in parsed):
                            return parsed
                    except:
                        try:
                            # Final fallback: split string
                            if embedding.startswith('[') and embedding.endswith(']'):
                                embedding = embedding[1:-1]
                            return [float(x) for x in embedding.split(',')]
                        except:
                            return None
            
            # If it's a list of non-float items, try to convert
            if isinstance(embedding, list):
                try:
                    return [float(x) for x in embedding]
                except:
                    return None
            
            # If it's a dict with 'embedding' key, use that
            if isinstance(embedding, dict) and 'embedding' in embedding:
                return parse_embedding(embedding['embedding'])
            
            return None

        # Embed documents asynchronously
        embeddings = await embedder_async.embed_documents_async(texts)
            
        # Ensure embeddings are properly formatted
        embeddings = [parse_embedding(embedding) for embedding in embeddings]
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk["text"])
            metadatas.append(chunk["metadata"])
            # Create unique ID based on content and metadata
            unique_str = f"{chunk['text']}{str(chunk['metadata'])}"
            ids.append(hashlib.sha256(unique_str.encode()).hexdigest())
            logger.debug(f"Created ID for chunk: {unique_str[:50]}...")
        
        # Filter out any None or empty embeddings
        valid_data = [
            (d, m, i, e) for d, m, i, e in zip(documents, metadatas, ids, embeddings)
            if e is not None and len(e) > 0
        ]
        
        # Unzip the valid data
        if valid_data:
            documents, metadatas, ids, embeddings = zip(*valid_data)
            
            # Add to collection
            try:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
                logger.info(f"Added {len(valid_data)} embeddings to collection")
            except Exception as e:
                logger.error(f"Error adding embeddings to collection: {e}")
                if embeddings:
                    first_embedding = embeddings[0]
                    logger.error(f"First embedding type: {type(first_embedding)}, length: {len(first_embedding)}")
        else:
            logger.warning("No valid embeddings to add to collection")

    async def process_repository(self, repo_url: str) -> List[Dict[str, Any]]:
        """Clone or update a repository and process all its files"""
        repo_path = await self.clone_or_update_repo(repo_url)
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        
        # Walk through all files in the repository
        all_chunks = []
        
        # Use asyncio.to_thread for file system operations
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_chunks = await self.process_file(file_path, repo_name)
                all_chunks.extend(file_chunks)
        
        logger.info(f"Processed repository {repo_url} with {len(all_chunks)} chunks")
        return all_chunks
    
    async def update(self):
        """Update all repositories and refresh the index asynchronously"""
        logger.info("Starting repository update...")
        all_chunks = []
        
        for repo_url in settings.REPOSITORIES:
            logger.info(f"Processing repository: {repo_url}")
            chunks = await self.process_repository(repo_url)
            all_chunks.extend(chunks)
            logger.info(f"Processed {len(chunks)} chunks from {repo_url}")
        
        # Create new index asynchronously
        logger.info("Creating new index...")
        await self.create_index(all_chunks, self.new_collection_name)
        
        # Atomically swap indexes
        logger.info("Swapping indexes...")
        collections = self.chroma_client.list_collections()
        if self.current_collection_name in [col.name for col in collections]:
            self.chroma_client.delete_collection(self.current_collection_name)
        self.chroma_client.get_collection(self.new_collection_name).modify(name=self.current_collection_name)
        
        logger.info("Index update complete")
    
    def get_query_collection(self):
        """Get the current collection for querying"""
        return self.chroma_client.get_collection(
            name=self.current_collection_name,
            embedding_function=self.ef
        )

# Singleton instance
github_sync = GitHubSync()