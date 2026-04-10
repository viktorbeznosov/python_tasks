# config.py

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    API_TOKEN: str
    LOG_PRINT: str = "1"
    JWT_SECRET: str
    JWT_EXPIRATION: int
    LLM_BASE_URL: str
    LLM_MODEL: str
    LLM_API_KEY: str
    HF_TOKEN: str
    WHISPER_MODEL: str
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"   
    )

settings = Settings()