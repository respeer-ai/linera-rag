import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_API_URL: str = "https://api.deepseek.com/v1"
    EMBEDDING_MODEL: str = "deepseek-embedding"
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