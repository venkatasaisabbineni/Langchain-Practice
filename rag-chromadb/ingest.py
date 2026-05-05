"""One-time ingestion pipeline for building a local Chroma vector store."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from utils import build_embeddings


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
DEFAULT_PDF_PATH = REPO_ROOT / "Data" / "bedrock-ug-2.pdf"
DEFAULT_DB_DIR = PROJECT_DIR / "vectorstore" / "chroma_db"
COLLECTION_NAME = "bedrock_pdf_chunks"
BATCH_SIZE = 64


def build_splitter() -> RecursiveCharacterTextSplitter:
    """Create chunker with the required defaults."""
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )


def chunk_pdf_lazily(pdf_path: Path, splitter: RecursiveCharacterTextSplitter) -> Iterable[Document]:
    """Yield chunked documents while streaming pages from the PDF."""
    loader = PyPDFLoader(str(pdf_path))
    for page_doc in loader.lazy_load():
        raw_page = int(page_doc.metadata.get("page", 0))
        normalized_page = raw_page + 1

        chunks = splitter.split_documents([page_doc])
        for chunk_index, chunk in enumerate(chunks):
            chunk.metadata["source"] = str(pdf_path)
            chunk.metadata["page"] = normalized_page
            chunk.metadata["chunk_index"] = chunk_index
            yield chunk


def reset_vectorstore_dir(persist_directory: Path) -> None:
    if persist_directory.exists():
        shutil.rmtree(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)


def ingest(pdf_path: Path, persist_directory: Path, force_reindex: bool) -> None:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    persist_directory.mkdir(parents=True, exist_ok=True)

    embeddings = build_embeddings()
    splitter = build_splitter()

    if force_reindex:
        print(f"[INFO] Reindex enabled, removing old DB at: {persist_directory}")
        reset_vectorstore_dir(persist_directory)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )

    existing_count = len(vectorstore.get()["ids"])
    if existing_count > 0 and not force_reindex:
        print("[INFO] Existing vectors found. Skipping ingestion.")
        print(f"[INFO] Existing chunk count: {existing_count}")
        print("[INFO] Use --force-reindex to rebuild from scratch.")
        return

    start = time.perf_counter()
    buffered_docs: list[Document] = []
    total_chunks = 0
    total_pages = set()

    for chunk in chunk_pdf_lazily(pdf_path=pdf_path, splitter=splitter):
        buffered_docs.append(chunk)
        total_chunks += 1
        total_pages.add(chunk.metadata["page"])

        if len(buffered_docs) >= BATCH_SIZE:
            vectorstore.add_documents(buffered_docs)
            buffered_docs.clear()

            if total_chunks % 1000 == 0:
                elapsed = time.perf_counter() - start
                print(f"[INFO] Processed {total_chunks} chunks in {elapsed:.1f}s")

    if buffered_docs:
        vectorstore.add_documents(buffered_docs)

    # Compatibility with older LangChain/Chroma integrations.
    if hasattr(vectorstore, "persist"):
        vectorstore.persist()

    elapsed_total = time.perf_counter() - start
    throughput = total_chunks / elapsed_total if elapsed_total > 0 else 0.0

    print("\n[INFO] Ingestion complete")
    print(f"[INFO] Pages processed: {len(total_pages)}")
    print(f"[INFO] Chunks embedded: {total_chunks}")
    print(f"[INFO] Elapsed time: {elapsed_total:.2f}s")
    print(f"[INFO] Throughput: {throughput:.2f} chunks/s")
    print(f"[INFO] Vector DB path: {persist_directory}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local Chroma vector store from PDF.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF_PATH, help="Path to source PDF.")
    parser.add_argument(
        "--persist-directory",
        type=Path,
        default=DEFAULT_DB_DIR,
        help="Directory for persisted Chroma files.",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Delete existing DB contents and rebuild index.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest(
        pdf_path=args.pdf,
        persist_directory=args.persist_directory,
        force_reindex=args.force_reindex,
    )
