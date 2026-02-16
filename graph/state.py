"""Shared state definition for the proposal agent graph."""

from __future__ import annotations
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State that flows through every node in the LangGraph pipeline.

    Attributes
    ----------
    messages : list[BaseMessage]
        Full conversation history (auto-merged via ``add_messages``).
    task : str
        The original user request / proposal brief.
    proposal_type : str
        Detected proposal category (Grant, Business, Technical, …).
    plan : str
        Structured plan produced by the Planner agent.
    research_data : str
        Contextual research gathered by the Research agent.
    search_queries : list[str]
        Search queries used by the Researcher agent (surfaced in UI).
    draft : str
        Current proposal draft.
    critique : str
        Evaluator feedback on the draft.
    score : float
        Numeric quality score assigned by the Evaluator.
    dimension_scores : dict
        Per-dimension scores from the Evaluator (Clarity, etc.).
    revision_count : int
        How many refinement iterations have occurred.
    user_feedback : str
        Input from user during human-in-the-loop steps.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    task: str
    proposal_type: str
    plan: str
    research_data: str
    search_queries: list[str]
    draft: str
    critique: str
    score: float
    dimension_scores: dict
    revision_count: int
    user_feedback: str
