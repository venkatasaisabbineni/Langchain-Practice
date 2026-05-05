"""Entry point for the multi-agent RAG system."""

from __future__ import annotations

import uuid
from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from graph import app


def run(query: str, session_id: str | None = None) -> str:
    """Invoke the multi-agent RAG graph and return the final answer."""
    if session_id is None:
        session_id = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    config: RunnableConfig = {
        "configurable": {
            "session_id": session_id,
            "thread_id": session_id,
        },
        "run_name": "multi-agent-rag",
        "tags": ["multi-agent-rag", "gemma3:12b", "chromadb"],
        "metadata": {"session_id": session_id},
    }

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "context_chunks": [],
        "revision_count": 0,
        "answer": "",
        "route": "",
    }

    final_state = app.invoke(initial_state, config=config)
    return final_state["answer"]


def main() -> None:
    print("Multi-Agent RAG (gemma3:12b + ChromaDB + LangGraph). Type 'quit' to exit.\n")
    session_id = f"session-{uuid.uuid4().hex[:8]}"

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        answer = run(query=query, session_id=session_id)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
