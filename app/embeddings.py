from openai import OpenAI
from app.config import settings

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
        return [embedding.embedding for embedding in response.data]
    
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query"""
        return self.embed_documents([text])[0]

# Singleton instance for easy access
embedder = DeepSeekEmbeddings()