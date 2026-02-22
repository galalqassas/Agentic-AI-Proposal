"""Integration test for the LangGraph pipeline (planner → researcher).

All LLM and Tavily calls are mocked so the test runs offline.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage

from agents.models import PlannerOutput, SearchQueries
from graph.graph import build_graph

# ── Mock outputs ─────────────────────────────────────────────────────
MOCK_PLANNER_OUTPUT = PlannerOutput(
    proposal_type="Technical",
    key_facts=["Client: TechCo", "Need: Cloud migration"],
    research_needed=[
        "TechCo company profile",
        "Cloud migration best practices 2026",
    ],
    proposal_sections=[
        "Executive Summary",
        "Current State Assessment",
        "Migration Strategy",
        "Timeline & Milestones",
        "Budget",
    ],
    questions_for_user=[],  # No questions — go straight to researcher
)

MOCK_PLANNER_WITH_QUESTIONS = PlannerOutput(
    proposal_type="Business",
    key_facts=["Industry: FinTech"],
    research_needed=["Market analysis"],
    proposal_sections=["Executive Summary", "Solution"],
    questions_for_user=["What is your company name?"],
)

MOCK_QUERIES = SearchQueries(
    queries=["TechCo company profile", "Cloud migration best practices 2026"]
)

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
        mock_p_structured = MagicMock()
        mock_p_structured.invoke.return_value = MOCK_PLANNER_OUTPUT
        mock_p_llm.with_structured_output.return_value = mock_p_structured
        mock_planner_llm_fn.return_value = mock_p_llm

        # ── Mock Researcher LLM ──
        mock_r_llm = MagicMock()
        mock_r_structured = MagicMock()
        mock_r_structured.invoke.return_value = MOCK_QUERIES
        mock_r_llm.with_structured_output.return_value = mock_r_structured
        mock_r_llm.invoke.return_value = AIMessage(content=MOCK_RESEARCH_BRIEF)
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
            "search_queries": [],
            "draft": "",
            "critique": "",
            "score": 0.0,
            "dimension_scores": {},
            "revision_count": 0,
            "user_feedback": "",
            "questions_for_user": [],
        }

        config = {"configurable": {"thread_id": "test-full-pipeline"}}
        result = graph.invoke(initial_state, config)

        # ── Assertions ──
        assert result["plan"] != ""
        assert "### Proposal Type" in result["plan"]
        assert result["proposal_type"] == "Technical"
        assert result["research_data"] != ""
        assert len(result["messages"]) >= 2  # planner + researcher msgs

    @patch("agents.planner.get_llm")
    def test_planner_questions_interrupt(self, mock_planner_llm_fn):
        """When planner has questions, graph should interrupt at ask_user."""

        mock_p_llm = MagicMock()
        mock_p_structured = MagicMock()
        mock_p_structured.invoke.return_value = MOCK_PLANNER_WITH_QUESTIONS
        mock_p_llm.with_structured_output.return_value = mock_p_structured
        mock_planner_llm_fn.return_value = mock_p_llm

        graph = build_graph()
        initial_state = {
            "messages": [],
            "task": "Write a business proposal for fintech",
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

        config = {"configurable": {"thread_id": "test-planner-questions"}}
        graph.invoke(initial_state, config)

        # Graph should be interrupted after ask_user — next pending is researcher
        state = graph.get_state(config)
        assert state.next  # Graph is paused (not finished)
        assert len(state.values.get("questions_for_user", [])) > 0
        assert "What is your company name?" in state.values["questions_for_user"]

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
        mock_p_structured = MagicMock()
        mock_p_structured.invoke.return_value = MOCK_PLANNER_OUTPUT
        mock_p_llm.with_structured_output.return_value = mock_p_structured
        mock_planner_llm_fn.return_value = mock_p_llm

        mock_r_llm = MagicMock()
        mock_r_structured = MagicMock()
        mock_r_structured.invoke.return_value = MOCK_QUERIES
        mock_r_llm.with_structured_output.return_value = mock_r_structured
        mock_r_llm.invoke.return_value = AIMessage(content=MOCK_RESEARCH_BRIEF)
        mock_researcher_llm_fn.return_value = mock_r_llm

        mock_tavily_search.return_value = "Raw"

        graph = build_graph()
        result = graph.invoke(
            {
                "messages": [],
                "task": "Test",
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
            },
            {"configurable": {"thread_id": "test-state-flow"}},
        )

        # The plan should be set by planner, then researcher uses it
        assert result["plan"] != ""
        assert result["research_data"] == MOCK_RESEARCH_BRIEF
