"""
config.py — Centralized configuration using Pydantic BaseSettings.
Reads from .env file. Single source of truth for all tuneable parameters.
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All runtime configuration. Override any field via .env or shell env vars."""

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_llm_model: str = Field(default="llama3.1")
    ollama_embed_model: str = Field(default="nomic-embed-text")

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_collection: str = Field(default="behavioral_frameworks")

    # ── Ingestion ─────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=1024)
    chunk_overlap: int = Field(default=50)
    top_k_retrieval: int = Field(default=5)

    # ── Paths ─────────────────────────────────────────────────────────────────
    pdf_dir: Path = Field(default=Path("./data/pdfs"))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Module-level singleton — import this everywhere
settings = Settings()
