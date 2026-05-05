"""Node implementations and shared state for the multi-agent RAG graph."""

from __future__ import annotations

from typing import Annotated

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import patch_config
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from tools import search_documents

OLLAMA_MODEL = "gemma3:12b"
POOR_RESULTS_THRESHOLD = 0.3
MAX_REVISIONS = 2


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context_chunks: list[Document]
    revision_count: int
    query: str
    answer: str
    route: str  # "search" | "direct" | "retry" | "answer" | "finalize"


# ── Supervisor ────────────────────────────────────────────────────────────────

_SUPERVISOR_SYSTEM = """\
You are a routing agent. Classify the user query into one of two categories:
- "search": the query asks about specific facts, AWS Bedrock features, technical details,
  or any topic that likely requires document retrieval.
- "direct": the query is a greeting, simple math, general knowledge, or does not require
  document retrieval.

Respond with ONLY the single word "search" or "direct". No explanation."""


def supervisor_node(state: AgentState, config: RunnableConfig) -> dict:
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.0)
    query = state["query"]

    verdict = llm.invoke(
        [SystemMessage(content=_SUPERVISOR_SYSTEM), HumanMessage(content=query)],
        config=patch_config(config, run_name="Supervisor"),
    ).content.strip().lower()

    if verdict not in {"search", "direct"}:
        verdict = "search"

    if verdict == "direct":
        answer = llm.invoke(
            [HumanMessage(content=query)],
            config=patch_config(config, run_name="Supervisor-DirectAnswer"),
        ).content.strip()
        return {
            "route": "direct",
            "answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    return {"route": "search"}


# ── Librarian ─────────────────────────────────────────────────────────────────

_REFORMULATE_SYSTEM = """\
The previous search returned low-relevance results.
Rewrite the following query to be more specific and likely to retrieve better results.
Return ONLY the rewritten query, no explanation."""


def librarian_node(state: AgentState, config: RunnableConfig) -> dict:
    query = state["query"]
    revision_count = state.get("revision_count", 0)

    raw_results: list[dict] = search_documents.invoke(
        {"query": query},
        config=patch_config(config, run_name="Librarian"),
    )

    docs = [
        Document(page_content=r["page_content"], metadata=r["metadata"])
        for r in raw_results
    ]
    scores = [r["relevance_score"] for r in raw_results]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    print(f"[Librarian] avg_relevance={avg_score:.3f}, revision_count={revision_count}")

    if avg_score < POOR_RESULTS_THRESHOLD and revision_count < MAX_REVISIONS:
        llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.3)
        new_query = llm.invoke(
            [SystemMessage(content=_REFORMULATE_SYSTEM), HumanMessage(content=query)],
            config=patch_config(config, run_name="Librarian-Reformulate"),
        ).content.strip()
        print(f"[Librarian] Reformulated query: {new_query!r}")
        return {
            "query": new_query,
            "context_chunks": docs,
            "revision_count": revision_count + 1,
            "route": "retry",
        }

    return {
        "context_chunks": docs,
        "revision_count": revision_count,
        "route": "answer",
    }


# ── Answerer ──────────────────────────────────────────────────────────────────

_ANSWER_SYSTEM = """\
You are a helpful assistant. Use ONLY the following retrieved context to answer the question.
If the answer cannot be found in the context, say "I don't know based on the provided document."
Keep answers concise and factual.

Context:
{context}"""


def answerer_node(state: AgentState, config: RunnableConfig) -> dict:
    query = state["query"]
    chunks: list[Document] = state.get("context_chunks", [])

    context_text = "\n\n---\n\n".join(
        f"[Page {c.metadata.get('page', '?')}]\n{c.page_content}" for c in chunks
    )

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.3)
    answer = llm.invoke(
        [
            SystemMessage(content=_ANSWER_SYSTEM.format(context=context_text)),
            HumanMessage(content=query),
        ],
        config=patch_config(config, run_name="Answerer"),
    ).content.strip()

    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


# ── Critic ────────────────────────────────────────────────────────────────────

_CRITIC_SYSTEM = """\
You are a factual grounding validator. Given a question, retrieved context chunks, and a
generated answer, determine if the answer is grounded in the context or contains hallucinations.

- "valid" if every factual claim in the answer can be traced to the context.
- "hallucinated" if the answer makes claims not supported by the context.

Respond with ONLY "valid" or "hallucinated". No explanation."""


def critic_node(state: AgentState, config: RunnableConfig) -> dict:
    query = state["query"]
    answer = state.get("answer", "")
    chunks: list[Document] = state.get("context_chunks", [])
    revision_count = state.get("revision_count", 0)

    context_text = "\n\n---\n\n".join(
        f"[Page {c.metadata.get('page', '?')}]\n{c.page_content}" for c in chunks
    )
    critic_prompt = (
        f"Question: {query}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Answer:\n{answer}"
    )

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.0)
    verdict = llm.invoke(
        [SystemMessage(content=_CRITIC_SYSTEM), HumanMessage(content=critic_prompt)],
        config=patch_config(config, run_name="Critic"),
    ).content.strip().lower()

    if verdict not in {"valid", "hallucinated"}:
        verdict = "valid"

    print(f"[Critic] verdict={verdict}, revision_count={revision_count}")

    if verdict == "hallucinated" and revision_count < MAX_REVISIONS:
        return {
            "revision_count": revision_count + 1,
            "route": "retry",
        }

    return {"route": "finalize"}
