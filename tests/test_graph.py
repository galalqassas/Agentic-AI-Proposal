"""Integration test for the LangGraph pipeline (planner → researcher).

All LLM and Tavily calls are mocked so the test runs offline.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage

from graph.graph import build_graph

# ── Mock responses ───────────────────────────────────────────────────
MOCK_PLANNER_OUTPUT = """\
### Proposal Type
Technical

### Key Facts Provided
- Client: TechCo
- Need: Cloud migration

### Research Needed
- TechCo company profile
- Cloud migration best practices 2026

### Proposal Plan
1. Executive Summary
2. Current State Assessment
3. Migration Strategy
4. Timeline & Milestones
5. Budget
"""

MOCK_QUERIES_OUTPUT = """\
1. TechCo company profile
2. Cloud migration best practices 2026
"""

MOCK_RESEARCH_BRIEF = """\
## Research Brief

### TechCo
TechCo is a mid-size SaaS provider based in Austin, TX.

### Cloud Migration Trends
80% of enterprises will migrate to cloud by 2027.
"""


class TestGraphIntegration:
    """End-to-end test of the planner → researcher pipeline."""

    @patch("agents.researcher._search_tavily")
    @patch("agents.researcher.get_llm")
    @patch("agents.planner.get_llm")
    def test_full_pipeline(
        self,
        mock_planner_llm_fn,
        mock_researcher_llm_fn,
        mock_tavily_search,
    ):
        """Graph should flow from planner to researcher and produce
        both a plan and research_data."""

        # ── Mock Planner LLM ──
        mock_p_llm = MagicMock()
        mock_p_llm.invoke.return_value = AIMessage(content=MOCK_PLANNER_OUTPUT)
        mock_planner_llm_fn.return_value = mock_p_llm

        # ── Mock Researcher LLM (called twice: queries + synthesis) ──
        mock_r_llm = MagicMock()
        mock_r_llm.invoke.side_effect = [
            AIMessage(content=MOCK_QUERIES_OUTPUT),
            AIMessage(content=MOCK_RESEARCH_BRIEF),
        ]
        mock_researcher_llm_fn.return_value = mock_r_llm

        # ── Mock Tavily ──
        mock_tavily_search.return_value = "Raw results"

        # ── Run graph ──
        graph = build_graph()
        initial_state = {
            "messages": [],
            "task": "Write a technical proposal for TechCo cloud migration",
            "proposal_type": "",
            "plan": "",
            "research_data": "",
            "draft": "",
            "critique": "",
            "score": 0.0,
            "revision_count": 0,
        }

        result = graph.invoke(initial_state)

        # ── Assertions ──
        assert result["plan"] != ""
        assert "### Proposal Type" in result["plan"]
        assert result["proposal_type"] == "Technical"
        assert result["research_data"] != ""
        assert len(result["messages"]) >= 2  # planner + researcher msgs

    @patch("agents.researcher._search_tavily")
    @patch("agents.researcher.get_llm")
    @patch("agents.planner.get_llm")
    def test_state_flows_between_nodes(
        self,
        mock_planner_llm_fn,
        mock_researcher_llm_fn,
        mock_tavily_search,
    ):
        """Verify planner output becomes researcher input."""

        mock_p_llm = MagicMock()
        mock_p_llm.invoke.return_value = AIMessage(content=MOCK_PLANNER_OUTPUT)
        mock_planner_llm_fn.return_value = mock_p_llm

        mock_r_llm = MagicMock()
        mock_r_llm.invoke.side_effect = [
            AIMessage(content=MOCK_QUERIES_OUTPUT),
            AIMessage(content=MOCK_RESEARCH_BRIEF),
        ]
        mock_researcher_llm_fn.return_value = mock_r_llm

        mock_tavily_search.return_value = "Raw"

        graph = build_graph()
        result = graph.invoke({
            "messages": [],
            "task": "Test",
            "proposal_type": "",
            "plan": "",
            "research_data": "",
            "draft": "",
            "critique": "",
            "score": 0.0,
            "revision_count": 0,
        })

        # The plan should be set by planner, then researcher uses it
        assert result["plan"] == MOCK_PLANNER_OUTPUT
        assert result["research_data"] == MOCK_RESEARCH_BRIEF
