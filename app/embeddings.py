from openai import OpenAI
import aiohttp
import asyncio
from app.config import settings, EmbeddingType

class DeepSeekEmbeddings:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_API_URL
        )
        self.model = settings.EMBEDDING_MODEL
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents"""
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        # Ensure the response is properly formatted as a list of lists of floats
        return [list(map(float, embedding.embedding)) for embedding in response.data]
    
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query"""
        return self.embed_documents([text])[0]

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
    embedder = DeepSeekEmbeddings()
    embedder_async = None
elif settings.EMBEDDING_TYPE == EmbeddingType.CHUTES:
    embedder = None
    embedder_async = ChutesEmbeddings()
else:
    raise ValueError(f"Unsupported embedding type: {settings.EMBEDDING_TYPE}")