from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    VLLM_SERVER_URL: str = "http://localhost:8000"
    REDIS_URL: str = "redis://localhost:6379"
    MAX_LOADED_LORAS: int = 32
    MAX_DOWNLOADED_LORAS: int = 128
    HF_CACHE_DIR: Optional[str] = None
    DEFAULT_HF_TOKEN: Optional[str] = None
    PROXY_PORT: int = 8080
    BASE_MODEL: str

    # Anything written in the .env is used
    class Config: 
        env_file = ".env"

settings = Settings()