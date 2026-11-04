"""
Shared pydantic models:
  - Structured extraction schema (disease / symptoms / tests / treatments)
  - FastAPI request/response contracts
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Information-extraction schema
# ---------------------------------------------------------------------------
class DiseaseRecord(BaseModel):
    """One structured record extracted from a source medical document."""

    disease: str = Field(..., description="Name of the disease or condition")
    symptoms: List[str] = Field(default_factory=list, description="Reported symptoms")
    diagnostic_tests: List[str] = Field(default_factory=list, description="Tests used to diagnose it")
    treatments: List[str] = Field(default_factory=list, description="Treatments / management options")
    source_document: str = Field(..., description="Filename the record was extracted from")
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Model's self-reported confidence, if provided"
    )


class ExtractionBatchResult(BaseModel):
    records: List[DiseaseRecord]
    documents_processed: int
    records_extracted: int
    failed_documents: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# API contracts
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class SourceCitation(BaseModel):
    source_document: str
    chunk_id: str
    snippet: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    citations: List[SourceCitation]
    disclaimer: str = (
        "This system provides general medical information retrieved from indexed "
        "documents. It is not a substitute for professional diagnosis. Consult a "
        "licensed clinician for medical decisions."
    )


class IngestResponse(BaseModel):
    chunks_indexed: int
    documents_indexed: int
    index_path: str


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    documents_indexed: int
