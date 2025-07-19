from openai import AsyncOpenAI
import aiohttp
import asyncio
from app.config import settings, EmbeddingType
from app.logger import logger

class DeepSeekEmbeddings:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_API_URL
        )
        self.model = settings.EMBEDDING_MODEL
    
    async def embed_documents_async(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously embed multiple documents"""
        logger.debug(f"Embedding {len(texts)} documents")
        response = await self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        # Ensure the response is properly formatted as a list of lists of floats
        embeddings = [list(map(float, embedding.embedding)) for embedding in response.data]
        logger.debug(f"Got {len(embeddings)} embeddings with size {len(embeddings[0])} each. First embedding: {embeddings[0][:16]}")
        return embeddings
    
    async def embed_query_async(self, text: str) -> list[float]:
        """Asynchronously embed a single query"""
        logger.debug(f"Embedding query: {text[:50]}...")
        result = (await self.embed_documents_async([text]))[0]
        logger.debug(f"Query embedding result length: {len(result)}. First 16 values: {result[:16]}")
        return result

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Synchronously embed multiple documents"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.embed_documents_async(texts))
    
    def embed_query(self, text: str) -> list[float]:
        """Synchronously embed a single query"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.embed_query_async(text))

class ChutesEmbeddings:
    def __init__(self):
        self.api_url = settings.CHUTES_API_URL
        self.api_key = settings.CHUTES_API_KEY
    
    async def embed_documents_async(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously embed multiple documents"""
        async with aiohttp.ClientSession() as session:
            tasks = [self._embed_text(session, text) for text in texts]
            return await asyncio.gather(*tasks)
    
    async def embed_query_async(self, text: str) -> list[float]:
        """Asynchronously embed a single query"""
        async with aiohttp.ClientSession() as session:
            return await self._embed_text(session, text)
    
    async def _embed_text(self, session: aiohttp.ClientSession, text: str) -> list[float]:
        """Send async request to Chutes embedding API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        logger.debug(f"Sending request to Chutes API: {text[:50]}...")
        async with session.post(
            self.api_url,
            headers=headers,
            json={"inputs": text}
        ) as response:
            response.raise_for_status()
            result = await response.json()
            logger.debug(f"Received Chutes API response: {str(result)[:16]}...")
            # Handle both list and dict responses
            if isinstance(result, list):
                logger.debug(f"Got list response with {len(result)} items")
                return result  # Return the list directly
            elif isinstance(result, dict):
                embedding = result.get('embedding', [])
                logger.debug(f"Got dict response with embedding of length {len(embedding)}")
                return embedding  # Extract embedding if available
            else:
                logger.warning("Got unexpected response type")
                return []  # Default to empty list for unexpected responses

# Choose embedder based on configuration
if settings.EMBEDDING_TYPE == EmbeddingType.DEEPSEEK:
    embedder_async = DeepSeekEmbeddings()
elif settings.EMBEDDING_TYPE == EmbeddingType.CHUTES:
    embedder_async = ChutesEmbeddings()
else:
    raise ValueError(f"Unsupported embedding type: {settings.EMBEDDING_TYPE}")