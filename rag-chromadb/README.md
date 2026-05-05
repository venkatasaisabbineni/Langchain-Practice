# Offline PDF RAG with Local ChromaDB

This project builds a fully offline Retrieval-Augmented Generation (RAG) index from:

- `Data/bedrock-ug-2.pdf`

It is optimized for Apple Silicon by running embeddings on `mps` (Metal Performance Shaders).

## Files

- `ingest.py`: one-time ingestion + chunking + embedding + Chroma persistence.
- `query.py`: semantic search over persisted vectors (`k=4`).
- `vectorstore/chroma_db`: on-disk Chroma data files.

## Install

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r rag-chromadb/requirements.txt
```

## Ingest (run once)

```bash
python rag-chromadb/ingest.py
```

Optional full rebuild:

```bash
python rag-chromadb/ingest.py --force-reindex
```

What ingestion does:

- Reads the large PDF lazily via `PyPDFLoader(...).lazy_load()`.
- Splits text with `RecursiveCharacterTextSplitter`:
  - `chunk_size=1000`
  - `chunk_overlap=150`
  - `separators=["\\n\\n", "\\n", " ", ""]`
- Embeds with `sentence-transformers/all-MiniLM-L6-v2` on `mps`.
- Persists vectors to `rag-chromadb/vectorstore/chroma_db`.
- Prints elapsed ingestion time and throughput.

## Query

```bash
python rag-chromadb/query.py --query "How does Bedrock knowledge base retrieval work?"
```

Or run interactive mode:

```bash
python rag-chromadb/query.py
```

Each result prints:

- rank
- similarity score
- exact source page number
- source PDF path
- chunk text content

## Notes

- No external API keys or cloud endpoints are used.
- If `mps` is unavailable on your runtime, install/update PyTorch for Apple Silicon.
