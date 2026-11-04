"""
Stage 1: Document Loader
=========================
Loads PDF / DOCX / TXT medical documents from a directory into LangChain
`Document` objects, tagging each with source metadata used later for
citations.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def _load_single_file(path: Path) -> List[Document]:
    """Load one file into one or more LangChain Documents (loader dependent)."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
    elif suffix == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported extension: {suffix}")

    docs = loader.load()
    for d in docs:
        d.metadata["source_document"] = path.name
    return docs


def load_documents(directory: Path) -> List[Document]:
    """
    Load every supported file in `directory` (non-recursive by default).

    Returns a flat list of LangChain Documents. Files that fail to load are
    logged and skipped rather than aborting the whole batch — a single
    corrupt PDF should not block ingestion of the other 99 files.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Document directory does not exist: {directory}")

    files = [p for p in sorted(directory.iterdir()) if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        logger.warning("No supported documents (%s) found in %s", SUPPORTED_EXTENSIONS, directory)
        return []

    all_docs: List[Document] = []
    for path in files:
        try:
            docs = _load_single_file(path)
            all_docs.extend(docs)
            logger.info("Loaded %s (%d page/section chunks)", path.name, len(docs))
        except Exception as exc:  # noqa: BLE001 - intentional broad catch for batch robustness
            logger.error("Failed to load %s: %s", path.name, exc)

    logger.info("Document loading complete: %d files -> %d raw documents", len(files), len(all_docs))
    return all_docs
