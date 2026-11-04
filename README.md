# Medical Diagnosis AI Assistance — RAG Pipeline

Production-structured implementation of:

```
MEDICAL DOCUMENTS
      │
      ▼
Document Loader (PDF/DOCX/TXT)
      │
      ├──► Information Extraction ──► Structured DB (JSON)
      │
      └──► Chunk + Metadata ──► Embeddings ──► FAISS ──► Retriever+Reranker
                                                                │
                                                                ▼
                                                        LLM / Agent Layer
                                                                │
                                                                ▼
                                                  FastAPI Medical Assistant
                                                                │
                                                                ▼
                                                     Answer + Citations
```

## Project layout

```
medical_rag_pipeline/
├── app/
│   ├── config.py            # env-driven settings (pydantic-settings)
│   ├── schemas.py           # extraction + API pydantic models
│   ├── document_loader.py   # Stage 1: PDF/DOCX/TXT ingestion
│   ├── extraction/
│   │   └── pipeline.py      # Stage 2: LLM structured extraction
│   ├── rag/
│   │   ├── ingest.py        # chunk -> embed -> FAISS build/load
│   │   ├── retriever.py     # FAISS similarity search + cross-encoder rerank
│   │   └── chain.py         # grounded answer generation + citations
│   └── main.py               # FastAPI app (/ingest /extract /chat /health)
├── data/
│   ├── raw_docs/             # <- put your source PDFs/DOCX/TXT here
│   ├── extracted/            # extracted_records.json output lands here
│   └── faiss_index/          # persisted vector index
├── requirements.txt
└── .env.example
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in ANTHROPIC_API_KEY
```

Drop your medical documents (PDF/DOCX/TXT) into `data/raw_docs/`.

## Execution flow

**1. Start the API**
```bash
uvicorn app.main:app --reload
```
On startup, the app tries to load an existing FAISS index from `data/faiss_index/`.
If none exists yet, `/chat` returns 503 until you run `/ingest`.

**2. Build the vector index** (Document Loader → Chunk+Metadata → Embeddings → FAISS)
```bash
curl -X POST http://localhost:8000/ingest
```
This loads every file in `data/raw_docs/`, splits it with `RecursiveCharacterTextSplitter`
(default 800 chars / 120 overlap), embeds chunks locally with
`sentence-transformers/all-MiniLM-L6-v2`, and persists a FAISS index to
`data/faiss_index/`.

**3. Run structured extraction** (Document Loader → Information Extraction → Structured DB)
```bash
curl -X POST http://localhost:8000/extract
```
Runs an LLM with `with_structured_output` over every loaded document to pull
`{disease, symptoms, diagnostic_tests, treatments, source_document}` records,
writing them to `data/extracted/extracted_records.json`. Failures on individual
documents are logged and skipped rather than aborting the batch — this is what
gives you the "processed 100+ documents, most auto-extracted" throughput claim
instead of a brittle all-or-nothing job.

**4. Ask questions** (Retriever+Reranker → LLM/Agent Layer → Answer+Citations)
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What diagnostic tests are used for type 2 diabetes?"}'
```
Response:
```json
{
  "answer": "Type 2 diabetes is typically diagnosed using fasting plasma glucose and HbA1c testing [1]...",
  "citations": [
    {"source_document": "endocrinology_guide.pdf", "chunk_id": "endocrinology_guide.pdf::chunk_14",
     "snippet": "...", "score": 5.82}
  ],
  "disclaimer": "This system provides general medical information ..."
}
```

**5. Check status**
```bash
curl http://localhost:8000/health
```

## Design decisions worth knowing for an interview walkthrough

- **Two-stage retrieval (FAISS + cross-encoder rerank).** FAISS bi-encoder
  search is fast but approximate over ~5,000+ chunks; a cross-encoder reads
  query and chunk together for a much sharper relevance signal, applied only
  to the top ~20 candidates to keep latency bounded.
- **Local embeddings/reranker, hosted LLM for generation.** Embedding and
  reranking run on `sentence-transformers` locally — no per-query API cost
  and no external latency for the retrieval hot path. The LLM (Claude) is
  reserved for extraction and final answer synthesis, where reasoning quality
  actually matters.
- **Structured output over regex parsing.** Extraction uses
  `with_structured_output` against a pydantic schema so malformed LLM output
  fails validation loudly instead of silently corrupting the structured DB.
- **Citations are grounded, not decorative.** The prompt requires inline
  `[n]` markers; the API only returns citation objects for markers that
  actually appear in the generated answer, so citations map 1:1 to claims.
- **Batch resilience.** Both document loading and extraction catch per-item
  failures and continue, logging and reporting a `failed_documents` list
  instead of crashing a 100-document job over one bad file.
- **Safety framing.** The system prompt and API response both carry an
  explicit non-diagnostic disclaimer — this is a retrieval/QA layer over
  reference material, not a clinical decision tool.

## Known extension points

- Swap `FAISS` for a managed store (pgvector, Pinecone) by changing only
  `app/rag/ingest.py` / `retriever.py` — the rest of the pipeline is
  vector-store agnostic.
- Add auth (API key / OAuth) at the FastAPI layer before any real deployment.
- Add a `/extract/status` endpoint backed by a task queue (Celery/RQ) if
  extraction batches grow large enough to need async processing instead of
  blocking the request.
