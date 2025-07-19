from openai import AsyncOpenAI
import aiohttp
import asyncio
from app.config import settings, EmbeddingType

class DeepSeekEmbeddings:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_API_URL
        )
        self.model = settings.EMBEDDING_MODEL
    
    async def embed_documents_async(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously embed multiple documents"""
        response = await self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        # Ensure the response is properly formatted as a list of lists of floats
        return [list(map(float, embedding.embedding)) for embedding in response.data]
    
    async def embed_query_async(self, text: str) -> list[float]:
        """Asynchronously embed a single query"""
        return (await self.embed_documents_async([text]))[0]

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
        async with session.post(
            self.api_url,
            headers=headers,
            json={"inputs": text}
        ) as response:
            response.raise_for_status()
            result = await response.json()
            # Handle both list and dict responses
            if isinstance(result, list):
                return result  # Return the list directly
            elif isinstance(result, dict):
                return result.get('embedding', [])  # Extract embedding if available
            else:
                return []  # Default to empty list for unexpected responses

# Choose embedder based on configuration
if settings.EMBEDDING_TYPE == EmbeddingType.DEEPSEEK:
    embedder_async = DeepSeekEmbeddings()
elif settings.EMBEDDING_TYPE == EmbeddingType.CHUTES:
    embedder_async = ChutesEmbeddings()
else:
    raise ValueError(f"Unsupported embedding type: {settings.EMBEDDING_TYPE}")

# For backward compatibility - synchronous wrapper around async embedder
class SyncEmbeddingWrapper:
    def __init__(self, async_embedder):
        self.async_embedder = async_embedder
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Synchronous wrapper for embed_documents_async"""
        return asyncio.run(self.async_embedder.embed_documents_async(texts))
    
    def embed_query(self, text: str) -> list[float]:
        """Synchronous wrapper for embed_query_async"""
        return asyncio.run(self.async_embedder.embed_query_async(text))

# Create a synchronous wrapper for backward compatibility
embedder = SyncEmbeddingWrapper(embedder_async)