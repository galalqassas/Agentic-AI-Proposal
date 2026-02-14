"""LangGraph graph construction.

Complete pipeline:
START → planner → researcher → [INTERRUPT] → writer → evaluator → [condition] → output → END
                                                ↑           ↓
                                                └───────────┘
                                              (refinement loop)
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import AgentState
from agents.planner import planner_node
from agents.researcher import researcher_node
from agents.writer import writer_node
from agents.evaluator import evaluator_node
from agents.output import output_node

load_dotenv()

# Configuration from .env
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", 5))
QUALITY_THRESHOLD = float(os.getenv("QUALITY_THRESHOLD", 8.0))

def route_evaluator(state: AgentState) -> str:
    """Determine next step based on score and iteration count."""
    score = state.get("score", 0.0)
    revisions = state.get("revision_count", 0)
    
    if score >= QUALITY_THRESHOLD or revisions >= MAX_ITERATIONS:
        return "output"
    return "writer"

def build_graph() -> StateGraph:
    """Build and compile the proposal agent graph."""
    graph = StateGraph(AgentState)

    # Checkpointer for HITL and state persistence
    checkpointer = MemorySaver()

    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("output", output_node)

    # Add edges
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "evaluator")
    
    # Conditional edge
    graph.add_conditional_edges(
        "evaluator",
        route_evaluator,
        {
            "output": "output",
            "writer": "writer"
        }
    )
    
    graph.add_edge("output", END)

    # Compile with checkpointer and interrupt after researcher for HITL
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_after=["researcher"]
    )
