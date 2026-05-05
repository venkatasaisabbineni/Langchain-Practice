"""Query local Chroma DB for semantic retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_community.vectorstores import Chroma

from utils import build_embeddings


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_DIR = PROJECT_DIR / "vectorstore" / "chroma_db"
COLLECTION_NAME = "bedrock_pdf_chunks"
TOP_K = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query local Chroma semantic index.")
    parser.add_argument(
        "--persist-directory",
        type=Path,
        default=DEFAULT_DB_DIR,
        help="Directory containing persisted Chroma DB.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Question to run against local vector DB.",
    )
    return parser.parse_args()


def run_query(query: str, persist_directory: Path) -> None:
    if not persist_directory.exists():
        raise FileNotFoundError(
            f"Persist directory not found: {persist_directory}. Run ingest.py first."
        )

    embeddings = build_embeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )

    scored_results = vectorstore.similarity_search_with_score(query, k=TOP_K)

    print(f"\nQuery: {query}")
    print(f"Top {TOP_K} semantic matches:\n")

    for idx, (doc, score) in enumerate(scored_results, start=1):
        metadata = doc.metadata or {}
        page = metadata.get("page", "unknown")
        source = metadata.get("source", "unknown")
        print(f"[{idx}] score={score:.6f} | page={page} | source={source}")
        print(doc.page_content.strip())
        print("-" * 80)


if __name__ == "__main__":
    args = parse_args()
    query_text = args.query or input("Enter your query: ").strip()

    if not query_text:
        raise ValueError("Query cannot be empty.")

    run_query(query=query_text, persist_directory=args.persist_directory)
