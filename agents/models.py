"""Pydantic models for structured LLM output across all agents.

These models are used with ``ChatGroq.with_structured_output()`` to
replace fragile regex-based parsing with validated, typed responses.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Planner ──────────────────────────────────────────────────────────
class PlannerOutput(BaseModel):
    """Structured output returned by the Planner agent."""

    proposal_type: str = Field(
        description=(
            "Detected proposal category. Must be one of: "
            "Grant, Business, Technical, Sales, Project, Research, "
            "Partnership, or General."
        )
    )
    key_facts: list[str] = Field(
        description="Key facts the user has already provided."
    )
    research_needed: list[str] = Field(
        description="Topics that need external research (web search)."
    )
    proposal_sections: list[str] = Field(
        description="Ordered list of section titles for the proposal."
    )
    questions_for_user: list[str] = Field(
        default_factory=list,
        description=(
            "Short, easy-to-answer questions for information that "
            "cannot be found via web search (e.g. sender company name, "
            "budget, internal deadlines). Leave empty if the user "
            "has provided enough context."
        ),
    )


# ── Researcher ───────────────────────────────────────────────────────
class SearchQueries(BaseModel):
    """Search queries extracted from the plan by the Researcher."""

    queries: list[str] = Field(
        description=(
            "Up to 5 specific, targeted search queries to gather "
            "data that will personalise and strengthen the proposal."
        )
    )


# ── Evaluator ────────────────────────────────────────────────────────
class EvaluationOutput(BaseModel):
    """Structured evaluation scores and critique from the Evaluator."""

    clarity: float = Field(description="Score 0-10: clear language, easy to read.")
    persuasiveness: float = Field(description="Score 0-10: compelling arguments, benefits-focused.")
    completeness: float = Field(description="Score 0-10: addresses all task requirements.")
    structure: float = Field(description="Score 0-10: logical flow, proper formatting.")
    specificity: float = Field(description="Score 0-10: customised to client/industry, not generic.")
    overall_score: float = Field(description="Average of the 5 dimension scores (0-10).")
    critique: str = Field(
        default="",
        description=(
            "If overall_score < 9.0, provide exactly 3 specific improvements. "
            "If overall_score >= 9.0, leave empty."
        ),
    )
