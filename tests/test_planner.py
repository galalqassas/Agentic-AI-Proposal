"""Tests for the Planner agent.

These tests mock the LLM so they run offline and fast.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage

from agents.planner import planner_node, _extract_proposal_type

# ── Sample LLM response ─────────────────────────────────────────────
SAMPLE_PLAN_OUTPUT = """\
### Proposal Type
Business

### Key Facts Provided
- Client: Acme Corp
- Project: AI consulting engagement
- Timeline: Q3 2026

### Research Needed
- Acme Corp company background and recent news
- AI consulting market size and growth trends
- Competitor pricing benchmarks

### Proposal Plan
1. Executive Summary
2. Company Background & Understanding
3. Problem Statement
4. Proposed Solution
5. Methodology & Timeline
6. Team & Qualifications
7. Budget & Pricing
8. Terms & Conditions
"""


# ── Unit tests ───────────────────────────────────────────────────────
class TestExtractProposalType:
    """Tests for the _extract_proposal_type helper."""

    def test_extracts_business(self):
        assert _extract_proposal_type(SAMPLE_PLAN_OUTPUT) == "Business"

    def test_extracts_grant(self):
        text = "### Proposal Type\nGrant\n\n### Key Facts"
        assert _extract_proposal_type(text) == "Grant"

    def test_extracts_technical(self):
        text = "### Proposal Type\nTechnical\n\n### Key Facts"
        assert _extract_proposal_type(text) == "Technical"

    def test_extracts_with_colon(self):
        """Should extract proposal type when separated by a colon."""
        plan = "### Proposal Type: Business\n### Key Facts..."
        assert _extract_proposal_type(plan) == "Business"

    def test_fallback_to_general(self):
        text = "No structured output here at all."
        assert _extract_proposal_type(text) == "General"

    def test_case_insensitive_heading(self):
        text = "### proposal type\nResearch\n\n### Key Facts"
        assert _extract_proposal_type(text) == "Research"


class TestPlannerNode:
    """Tests for the planner_node graph function."""

    @patch("agents.planner.get_llm")
    def test_returns_plan_in_state(self, mock_get_llm):
        """planner_node should return plan and proposal_type in state."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content=SAMPLE_PLAN_OUTPUT)
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [],
            "task": "Write a proposal for AI consulting for Acme Corp",
            "proposal_type": "",
            "plan": "",
            "research_data": "",
            "draft": "",
            "critique": "",
            "score": 0.0,
            "revision_count": 0,
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
    def test_plan_contains_required_sections(self, mock_get_llm):
        """The plan should contain all key sections."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content=SAMPLE_PLAN_OUTPUT)
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [],
            "task": "Write a business proposal",
            "proposal_type": "",
            "plan": "",
            "research_data": "",
            "draft": "",
            "critique": "",
            "score": 0.0,
            "revision_count": 0,
        }

        result = planner_node(state)
        plan = result["plan"]

        assert "### Proposal Type" in plan
        assert "### Key Facts Provided" in plan
        assert "### Research Needed" in plan
        assert "### Proposal Plan" in plan

    @patch("agents.planner.get_llm")
    def test_calls_llm_with_task(self, mock_get_llm):
        """Verify the LLM is invoked with formatted messages."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="### Proposal Type\nGeneral")
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [],
            "task": "Test task",
            "proposal_type": "",
            "plan": "",
            "research_data": "",
            "draft": "",
            "critique": "",
            "score": 0.0,
            "revision_count": 0,
        }

        planner_node(state)

        mock_llm.invoke.assert_called_once()
        # The first call's first arg should be a list of messages
        call_args = mock_llm.invoke.call_args[0][0]
        assert len(call_args) >= 2  # system + human
