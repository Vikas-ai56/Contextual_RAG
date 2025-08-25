from pydantic_settings import BaseSettings
from functools import lru_cache
import logging
import sys
from dotenv import load_dotenv
import os

load_dotenv()

@lru_cache
def get_settings():
    return Settings()


class DocumentIntelligenceSettings(BaseSettings):
    api_key: str = os.getenv("DOCUMENT_INTELLIGENCE_API_KEY")
    endpoint: str = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")

class LLMSettings(BaseSettings):
    api_key: str = os.getenv("GOOGLE_API_KEY")
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

class CohereSettings(BaseSettings):
    api_key: str = os.getenv("COHERE_API_KEY")
    base_url: str = "https://api.cohere.ai/v2"
    client_name: str = "Development_Phase"
    timeout: float = 4.0

class ChromaDbSettings(BaseSettings):
    tenant: str = os.getenv("CHROMA_TENANT")
    database: str = os.getenv("CHROMA_DATABASE")
    token: str = os.getenv("CHROMA_TOKEN")

class Settings(BaseSettings):
    document_intelligence: DocumentIntelligenceSettings = DocumentIntelligenceSettings()
    llm: LLMSettings = LLMSettings()
    cohere: CohereSettings = CohereSettings()
    chromadb: ChromaDbSettings = ChromaDbSettings()