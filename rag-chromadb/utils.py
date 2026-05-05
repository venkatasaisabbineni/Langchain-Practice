"""Shared utilities for ingest, query, and chat scripts."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

_UTILS_DIR = Path(__file__).resolve().parent
load_dotenv(_UTILS_DIR / ".env")
load_dotenv()
if _hf_key := os.getenv("HUGGINGFACE_API_KEY"):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = _hf_key


def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "mps"},
        encode_kwargs={"device": "mps"},
    )
