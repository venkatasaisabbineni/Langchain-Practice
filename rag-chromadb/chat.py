"""RAG chatbot using local Ollama (gemma3:12b) and ChromaDB."""

from __future__ import annotations
from datetime import datetime
from pathlib import Path

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama

from utils import build_embeddings

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_DIR = PROJECT_DIR / "vectorstore" / "chroma_db"
COLLECTION_NAME = "bedrock_pdf_chunks"
OLLAMA_MODEL = "gemma3:12b"
TOP_K = 50


def build_chain(persist_directory: Path):
    if not persist_directory.exists():
        raise FileNotFoundError(
            f"Vector store not found at {persist_directory}. Run ingest.py first."
        )

    embeddings = build_embeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    doc_count = len(vectorstore.get()["ids"])
    print(f"[INFO] Connected to ChromaDB — {doc_count} chunks loaded.")

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.3)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})

    # Rephrases the user's follow-up questions into standalone queries using history.
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Given the chat history and the latest user question, reformulate the question "
            "to be fully standalone — no references to prior context. "
            "Return it as-is if it is already standalone. Do not answer it."
        )),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful assistant. Use ONLY the following retrieved context to answer. "
            "If the answer is not in the context, say 'I don't know based on the provided document.' "
            "Keep answers concise and factual.\n\n"
            "Context:\n{context}"
        )),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    answer_chain = create_stuff_documents_chain(llm, rag_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, answer_chain)

    session_history: dict[str, ChatMessageHistory] = {}

    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in session_history:
            session_history[session_id] = ChatMessageHistory()
        return session_history[session_id]

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


def main() -> None:
    chain = build_chain(DEFAULT_DB_DIR)
    session_id = str(datetime.now().strftime("%d-%m-%Y-%H-%M"))

    print(f"\nChatbot ready (model: {OLLAMA_MODEL}). Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        print("\nAssistant: ", end="", flush=True)
        context_docs = []
        stream_config = {
            "configurable": {"session_id": session_id},
            "run_name": "rag-chromadb-chat",
            "tags": ["rag-chromadb", "ollama", OLLAMA_MODEL.replace(":", "_")],
            "metadata": {
                "session_id": session_id,
                "ollama_model": OLLAMA_MODEL,
            },
        }
        for chunk in chain.stream({"input": user_input}, config=stream_config):
            if "context" in chunk:
                context_docs = chunk["context"]
            if "answer" in chunk:
                print(chunk["answer"], end="", flush=True)
        print("\n")

        if context_docs:
            pages = sorted({doc.metadata.get("page", "?") for doc in context_docs})
            print(f"Sources: {', '.join(f'p.{p}' for p in pages)}\n")


if __name__ == "__main__":
    main()
