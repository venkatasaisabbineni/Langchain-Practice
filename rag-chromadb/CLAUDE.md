# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The venv is at `.venv/` inside the project directory. Always activate it before running scripts.

## Environment

Create a `.env` file in **`rag-chromadb/`** next to `utils.py` with:
```
HUGGINGFACE_API_KEY=<your key>
```
All three scripts load this automatically via `python-dotenv`; [utils.py](utils.py) loads `.env` in this folder first, then any `.env` in the current working directory.

**LangSmith (optional tracing for `chat.py`):** add your Smith credentials so LangChain can submit runs:
```
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=<your LangSmith API key>
LANGSMITH_PROJECT=<project name>
```
Requires the `langsmith` package (`pip install -r requirements.txt`). Runs appear under `LANGSMITH_PROJECT` when `chat.py` invokes the chain.

## Running the pipeline

**Ingest** (run once; skips automatically if vectors already exist):
```bash
python ingest.py
python ingest.py --force-reindex   # wipe and rebuild
python ingest.py --pdf <path> --persist-directory <dir>
```

**Query** (one-shot retrieval, no LLM):
```bash
python query.py --query "How does Bedrock knowledge base retrieval work?"
python query.py   # interactive prompt
python query.py --persist-directory <dir>
```

**Chat** (conversational RAG with gemma3:12b via Ollama):
```bash
python chat.py
```
Requires Ollama running locally with `gemma3:12b` pulled (`ollama pull gemma3:12b`).

## Architecture

**`ingest.py`** — one-time build of the vector store.

**`ingest.py`** — one-time build of the vector store:
- Loads a PDF lazily page-by-page with `PyPDFLoader(...).lazy_load()`
- Splits into 1000-char chunks (150-char overlap) via `RecursiveCharacterTextSplitter`
- Embeds with `sentence-transformers/all-MiniLM-L6-v2` on `mps` (Apple Silicon)
- Writes to a persistent ChromaDB at `vectorstore/chroma_db/`, collection `bedrock_pdf_chunks`
- Batches `add_documents` calls in groups of 64 to bound memory

**`query.py`** — raw semantic retrieval (no LLM):
- Loads the same embedding model and connects to the same persisted Chroma collection
- Returns top-50 chunks by cosine similarity with scores, page numbers, and source path

**`chat.py`** — conversational RAG chatbot (LangChain v0.3 LCEL):
- `create_history_aware_retriever` — rephrases follow-up questions into standalone queries using session chat history before retrieval
- `create_stuff_documents_chain` — stuffs retrieved chunks into a system prompt and calls `ChatOllama(gemma3:12b, temperature=0.3)`
- `create_retrieval_chain` — wires both into a single invokable chain
- `RunnableWithMessageHistory` — manages per-session `ChatMessageHistory` automatically
- History is in-memory only; restarting `chat.py` starts a fresh session

**Data paths** (relative to the repo root, one level above this directory):
- Source PDF: `../Data/bedrock-ug-2.pdf`
- Vector store: `vectorstore/chroma_db/`

## Key constants

| Symbol | Value | Location |
|---|---|---|
| `COLLECTION_NAME` | `"bedrock_pdf_chunks"` | all scripts |
| `BATCH_SIZE` | `64` | `ingest.py` |
| `TOP_K` | `50` | `query.py`, `chat.py` |
| `OLLAMA_MODEL` | `"gemma3:12b"` | `chat.py` |
| chunk size / overlap | `1000` / `150` | `ingest.py` |

## Apple Silicon note

All scripts hard-code `device="mps"` in `model_kwargs` and `encode_kwargs`. On non-Mac hardware, change these to `"cpu"` or `"cuda"`.
