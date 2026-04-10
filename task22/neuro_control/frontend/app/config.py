# config.py

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    API_URL: str
    API_TOKEN: str
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"   
    )

settings = Settings()