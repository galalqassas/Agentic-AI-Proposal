"""Tests for the Planner agent.

These tests mock the LLM so they run offline and fast.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage

from agents.planner import planner_node, _format_plan
from agents.models import PlannerOutput

# ── Sample Pydantic output ───────────────────────────────────────────
SAMPLE_PLANNER_OUTPUT = PlannerOutput(
    proposal_type="Business",
    key_facts=[
        "Client: Acme Corp",
        "Project: AI consulting engagement",
        "Timeline: Q3 2026",
    ],
    research_needed=[
        "Acme Corp company background and recent news",
        "AI consulting market size and growth trends",
        "Competitor pricing benchmarks",
    ],
    proposal_sections=[
        "Executive Summary",
        "Company Background & Understanding",
        "Problem Statement",
        "Proposed Solution",
        "Methodology & Timeline",
        "Team & Qualifications",
        "Budget & Pricing",
        "Terms & Conditions",
    ],
    questions_for_user=[
        "What is the name of your company?",
        "Do you have a specific budget range in mind?",
    ],
)


# ── Unit tests ───────────────────────────────────────────────────────
class TestFormatPlan:
    """Tests for the _format_plan helper."""

    def test_contains_proposal_type(self):
        plan = _format_plan(SAMPLE_PLANNER_OUTPUT)
        assert "### Proposal Type" in plan
        assert "Business" in plan

    def test_contains_key_facts(self):
        plan = _format_plan(SAMPLE_PLANNER_OUTPUT)
        assert "### Key Facts Provided" in plan
        assert "Acme Corp" in plan

    def test_contains_research_needed(self):
        plan = _format_plan(SAMPLE_PLANNER_OUTPUT)
        assert "### Research Needed" in plan
        assert "AI consulting market" in plan

    def test_contains_proposal_sections(self):
        plan = _format_plan(SAMPLE_PLANNER_OUTPUT)
        assert "### Proposal Plan" in plan
        assert "Executive Summary" in plan

    def test_empty_key_facts(self):
        output = PlannerOutput(
            proposal_type="General",
            key_facts=[],
            research_needed=["topic"],
            proposal_sections=["Section 1"],
            questions_for_user=[],
        )
        plan = _format_plan(output)
        assert "(none provided)" in plan


class TestPlannerNode:
    """Tests for the planner_node graph function."""

    @patch("agents.planner.get_llm")
    def test_returns_plan_in_state(self, mock_get_llm):
        """planner_node should return plan and proposal_type in state."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = SAMPLE_PLANNER_OUTPUT
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [],
            "task": "Write a proposal for AI consulting for Acme Corp",
            "proposal_type": "",
            "plan": "",
            "research_data": "",
            "search_queries": [],
            "draft": "",
            "critique": "",
            "score": 0.0,
            "dimension_scores": {},
            "revision_count": 0,
            "user_feedback": "",
            "questions_for_user": [],
        }

        result = planner_node(state)

        assert "plan" in result
        assert len(result["plan"]) > 0
        assert "proposal_type" in result
        assert result["proposal_type"] == "Business"
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)

    @patch("agents.planner.get_llm")
    def test_returns_questions_for_user(self, mock_get_llm):
        """planner_node should return questions_for_user from structured output."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = SAMPLE_PLANNER_OUTPUT
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [],
            "task": "Write a proposal for AI consulting",
            "proposal_type": "",
            "plan": "",
            "research_data": "",
            "search_queries": [],
            "draft": "",
            "critique": "",
            "score": 0.0,
            "dimension_scores": {},
            "revision_count": 0,
            "user_feedback": "",
            "questions_for_user": [],
        }

        result = planner_node(state)

        assert "questions_for_user" in result
        assert len(result["questions_for_user"]) == 2
        assert "company" in result["questions_for_user"][0].lower()

    @patch("agents.planner.get_llm")
    def test_no_questions_when_info_sufficient(self, mock_get_llm):
        """planner_node should return empty questions when info is complete."""
        output_no_questions = PlannerOutput(
            proposal_type="Technical",
            key_facts=["Client: TechCo", "Budget: $100k"],
            research_needed=["TechCo background"],
            proposal_sections=["Executive Summary", "Solution"],
            questions_for_user=[],
        )
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = output_no_questions
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [],
            "task": "Write a technical proposal for TechCo, budget $100k",
            "proposal_type": "",
            "plan": "",
            "research_data": "",
            "search_queries": [],
            "draft": "",
            "critique": "",
            "score": 0.0,
            "dimension_scores": {},
            "revision_count": 0,
            "user_feedback": "",
            "questions_for_user": [],
        }

        result = planner_node(state)

        assert result["questions_for_user"] == []

    @patch("agents.planner.get_llm")
    def test_plan_contains_required_sections(self, mock_get_llm):
        """The plan should contain all key sections."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = SAMPLE_PLANNER_OUTPUT
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [],
            "task": "Write a business proposal",
            "proposal_type": "",
            "plan": "",
            "research_data": "",
            "search_queries": [],
            "draft": "",
            "critique": "",
            "score": 0.0,
            "dimension_scores": {},
            "revision_count": 0,
            "user_feedback": "",
            "questions_for_user": [],
        }

        result = planner_node(state)
        plan = result["plan"]

        assert "### Proposal Type" in plan
        assert "### Key Facts Provided" in plan
        assert "### Research Needed" in plan
        assert "### Proposal Plan" in plan

    @patch("agents.planner.get_llm")
    def test_calls_llm_with_structured_output(self, mock_get_llm):
        """Verify the LLM is invoked via with_structured_output."""
        output = PlannerOutput(
            proposal_type="General",
            key_facts=[],
            research_needed=[],
            proposal_sections=["Summary"],
            questions_for_user=[],
        )
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = output
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [],
            "task": "Test task",
            "proposal_type": "",
            "plan": "",
            "research_data": "",
            "search_queries": [],
            "draft": "",
            "critique": "",
            "score": 0.0,
            "dimension_scores": {},
            "revision_count": 0,
            "user_feedback": "",
            "questions_for_user": [],
        }

        planner_node(state)

        mock_llm.with_structured_output.assert_called_once_with(PlannerOutput)
        mock_structured.invoke.assert_called_once()
