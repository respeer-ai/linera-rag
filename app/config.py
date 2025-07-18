import os
from pydantic_settings import BaseSettings
from enum import Enum

class EmbeddingType(str, Enum):
    DEEPSEEK = "deepseek"
    CHUTES = "chutes"

class Settings(BaseSettings):
    # DeepSeek configuration
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_API_URL: str = "https://api.deepseek.com/v1"
    EMBEDDING_MODEL: str = "deepseek-embedding"
    
    # Chutes Embedding configuration
    CHUTES_API_URL: str = "https://chutes-baai-bge-large-en-v1-5.chutes.ai/embed"
    CHUTES_API_KEY: str = os.getenv("CHUTES_API_KEY", "")
    
    # Embedding type selection
    EMBEDDING_TYPE: EmbeddingType = EmbeddingType.CHUTES
    
    # Other settings
    REPOSITORIES: list[str] = [
        "https://github.com/linera-io/linera-protocol",
        "https://github.com/linera-io/linera-documentation"
    ]
    UPDATE_INTERVAL_HOURS: int = 6
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    DATA_DIR: str = "data"
    REPOS_DIR: str = "data/repos"
    CHROMA_DIR: str = "data/chroma_db"

    class Config:
        env_file = ".env"

settings = Settings()