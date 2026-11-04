"""
Centralized application configuration.

All tunables (model names, paths, chunking/retrieval params) live here and are
sourced from environment variables / .env so nothing is hardcoded in the
pipeline modules. Import `settings` anywhere you need config.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM ---
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.0

    # --- Embeddings / reranking (local models, no external API cost) ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Paths ---
    raw_docs_dir: Path = Path("data/raw_docs")
    extracted_dir: Path = Path("data/extracted")
    faiss_index_dir: Path = Path("data/faiss_index")

    # --- Chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- Retrieval ---
    top_k_retrieve: int = 20
    top_k_rerank: int = 5

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    def ensure_dirs(self) -> None:
        for d in (self.raw_docs_dir, self.extracted_dir, self.faiss_index_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
