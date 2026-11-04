"""
FastAPI Medical Assistant
==========================
Exposes:
  POST /ingest   - (re)build the FAISS index from data/raw_docs
  POST /extract  - run structured disease/symptom/test/treatment extraction
  POST /chat     - RAG question answering with source citations
  GET  /health   - liveness + index status

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from langchain_community.vectorstores import FAISS

from app.config import settings
from app.document_loader import load_documents
from app.extraction.pipeline import run_extraction
from app.rag.chain import generate_answer
from app.rag.ingest import build_faiss_index, chunk_documents, load_faiss_index
from app.rag.retriever import retrieve_and_rerank
from app.schemas import ChatRequest, ChatResponse, ExtractionBatchResult, HealthResponse, IngestResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_state: dict = {"vectorstore": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Try to load an existing index on startup so the API is immediately
    # queryable without forcing a rebuild every restart. If none exists yet,
    # /chat will 503 until /ingest is called.
    try:
        _state["vectorstore"] = load_faiss_index()
        logger.info("Loaded existing FAISS index at startup.")
    except FileNotFoundError:
        logger.warning("No FAISS index found at startup. Call POST /ingest before using /chat.")
        _state["vectorstore"] = None
    yield
    _state.clear()


app = FastAPI(
    title="Medical Diagnosis Assistance API",
    description=(
        "RAG-based medical information assistant. Provides general information "
        "retrieved from indexed documents with source citations. Not a diagnostic tool."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _vectorstore() -> FAISS:
    vs = _state.get("vectorstore")
    if vs is None:
        raise HTTPException(
            status_code=503,
            detail="No index loaded yet. Call POST /ingest first (or place documents in data/raw_docs).",
        )
    return vs


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    vs = _state.get("vectorstore")
    doc_count = vs.index.ntotal if vs is not None else 0
    return HealthResponse(status="ok", index_loaded=vs is not None, documents_indexed=doc_count)


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    """Load raw documents, chunk them, embed, and (re)build the FAISS index."""
    documents = load_documents(settings.raw_docs_dir)
    if not documents:
        raise HTTPException(
            status_code=400,
            detail=f"No supported documents found in {settings.raw_docs_dir} (.pdf/.docx/.txt).",
        )
    chunks = chunk_documents(documents)
    vectorstore = build_faiss_index(chunks)
    _state["vectorstore"] = vectorstore

    return IngestResponse(
        chunks_indexed=len(chunks),
        documents_indexed=len(documents),
        index_path=str(settings.faiss_index_dir),
    )


@app.post("/extract", response_model=ExtractionBatchResult)
def extract() -> ExtractionBatchResult:
    """Run LLM-based structured extraction over all documents in data/raw_docs."""
    documents = load_documents(settings.raw_docs_dir)
    if not documents:
        raise HTTPException(status_code=400, detail=f"No documents found in {settings.raw_docs_dir}.")
    return run_extraction(documents)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a question using RAG over the indexed medical documents."""
    vectorstore = _vectorstore()
    try:
        ranked_docs = retrieve_and_rerank(
            query=request.question,
            vectorstore=vectorstore,
            top_k_rerank=request.top_k,
        )
        answer_text, citations = generate_answer(request.question, ranked_docs)
        return ChatResponse(answer=answer_text, citations=citations)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat request failed")
        raise HTTPException(status_code=500, detail=f"Internal error while answering: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port, reload=True)
