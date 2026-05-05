"""ChromaDB retrieval tool for the multi-agent RAG system."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.tools import tool

from utils import build_embeddings

_PROJECT_DIR = Path(__file__).resolve().parent
VECTORSTORE_PATH = _PROJECT_DIR.parent / "rag-chromadb" / "vectorstore" / "chroma_db"
COLLECTION_NAME = "bedrock_pdf_chunks"
TOP_K = 50


@lru_cache(maxsize=1)
def _get_vectorstore() -> Chroma:
    if not VECTORSTORE_PATH.exists():
        raise FileNotFoundError(
            f"Vectorstore not found at {VECTORSTORE_PATH}. "
            "Run ingest.py in rag-chromadb first."
        )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=build_embeddings(),
        persist_directory=str(VECTORSTORE_PATH),
    )


@tool
def search_documents(query: str) -> list[dict[str, Any]]:
    """Search the AWS Bedrock documentation vectorstore for chunks relevant to the query.

    Returns up to 50 chunks with relevance scores where 0=irrelevant and 1=perfect match.
    """
    vs = _get_vectorstore()
    results = vs.similarity_search_with_relevance_scores(query, k=TOP_K)
    return [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
            "relevance_score": float(score),
        }
        for doc, score in results
    ]
