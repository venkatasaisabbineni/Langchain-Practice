"""LangGraph StateGraph definition for the multi-agent RAG system."""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from nodes import AgentState, answerer_node, critic_node, librarian_node, supervisor_node


def _route_after_supervisor(state: AgentState) -> Literal["librarian", "__end__"]:
    if state["route"] == "search":
        return "librarian"
    return END


def _route_after_librarian(state: AgentState) -> Literal["librarian", "answerer"]:
    if state["route"] == "retry":
        return "librarian"
    return "answerer"


def _route_after_critic(state: AgentState) -> Literal["librarian", "__end__"]:
    if state["route"] == "retry":
        return "librarian"
    return END


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("librarian", librarian_node)
    graph.add_node("answerer", answerer_node)
    graph.add_node("critic", critic_node)

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {"librarian": "librarian", END: END},
    )
    graph.add_conditional_edges(
        "librarian",
        _route_after_librarian,
        {"librarian": "librarian", "answerer": "answerer"},
    )
    graph.add_edge("answerer", "critic")
    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"librarian": "librarian", END: END},
    )

    return graph


app = build_graph().compile()
