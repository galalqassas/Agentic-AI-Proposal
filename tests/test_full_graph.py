"""End-to-end integration tests for the full proposal pipeline.

Tests the loop mechanism and final output generation with mocked agents.
"""

from __future__ import annotations

import os
import shutil
import pytest
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage
from graph.graph import build_graph
from agents.output import OUTPUT_DIR
from agents.models import PlannerOutput, SearchQueries, EvaluationOutput

# ── Mock data ────────────────────────────────────────────────────────
MOCK_PLANNER = PlannerOutput(
    proposal_type="Business",
    key_facts=["Client: TestCo"],
    research_needed=["TestCo background"],
    proposal_sections=["Executive Summary", "Solution"],
    questions_for_user=[],
)

MOCK_QUERIES = SearchQueries(queries=["TestCo background"])
MOCK_BRIEF = "Research brief"
MOCK_DRAFT_V1 = "Draft V1"
MOCK_DRAFT_V2 = "Draft V2"

MOCK_EVAL_LOW = EvaluationOutput(
    clarity=5.0,
    persuasiveness=5.0,
    completeness=5.0,
    structure=5.0,
    specificity=5.0,
    overall_score=5.0,
    critique="Improve content.",
)

MOCK_EVAL_HIGH = EvaluationOutput(
    clarity=9.8,
    persuasiveness=9.8,
    completeness=9.8,
    structure=9.8,
    specificity=9.8,
    overall_score=9.8,
    critique="",
)


class TestFullPipeline:
    """Tests the complete graph execution including interrupt/resume."""

    def setup_method(self):
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)

    def teardown_method(self):
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)

    @patch("agents.evaluator.get_llm")
    @patch("agents.writer.get_llm")
    @patch("agents.researcher._search_tavily")
    @patch("agents.researcher.get_llm")
    @patch("agents.planner.get_llm")
    def test_refinement_loop(
        self,
        mock_planner_llm,
        mock_researcher_llm,
        mock_tavily,
        mock_writer_llm,
        mock_evaluator_llm,
    ):
        """Should loop once (writer -> evaluator -> writer -> evaluator -> output).

        The graph interrupts after researcher, so we need to invoke twice:
        first to reach the interrupt, then resume to complete the pipeline.
        """

        # 1. Planner (structured output)
        mock_p_structured = MagicMock()
        mock_p_structured.invoke.return_value = MOCK_PLANNER
        mock_planner_llm.return_value.with_structured_output.return_value = mock_p_structured

        # 2. Researcher (structured queries + free-form synthesis)
        mock_r_structured = MagicMock()
        mock_r_structured.invoke.return_value = MOCK_QUERIES
        mock_researcher_llm.return_value.with_structured_output.return_value = mock_r_structured
        mock_researcher_llm.return_value.invoke.return_value = AIMessage(content=MOCK_BRIEF)
        mock_tavily.return_value = "Raw results"

        # 3. Writer (called twice: initial + refinement)
        mock_writer_llm.return_value.invoke.side_effect = [
            AIMessage(content=MOCK_DRAFT_V1),
            AIMessage(content=MOCK_DRAFT_V2),
        ]

        # 4. Evaluator (structured output, called twice: low score -> high score)
        mock_e_structured = MagicMock()
        mock_e_structured.invoke.side_effect = [MOCK_EVAL_LOW, MOCK_EVAL_HIGH]
        mock_evaluator_llm.return_value.with_structured_output.return_value = mock_e_structured

        app = build_graph()
        config = {"configurable": {"thread_id": "test-refinement"}}

        initial_state = {
            "messages": [],
            "task": "Write a proposal",
            "proposal_type": "",
            "plan": "",
            "research_data": "",
            "search_queries": [],
            "draft": "",
            "critique": "",
            "score": 0.0,
            "dimension_scores": {},
            "revision_count": 0,
            "questions_for_user": [],
        }

        # First invoke: runs planner -> researcher, then hits interrupt
        result = app.invoke(initial_state, config)

        # Verify we reached the interrupt (researcher completed)
        state = app.get_state(config)
        assert state.next  # Graph is paused

        # Resume: simulate user approving with "Proceed"
        app.update_state(config, {"user_feedback": "Proceed"}, as_node="researcher")
        result = app.invoke(None, config)

        # Assertions
        assert result["score"] == 9.8
        assert result["revision_count"] == 2  # 0->1 (low) -> 2 (high)
        assert result["draft"] == MOCK_DRAFT_V2

        # Verify loop execution
        assert mock_writer_llm.return_value.invoke.call_count == 2
        assert mock_e_structured.invoke.call_count == 2

        # Verify output creation
        assert os.path.exists(OUTPUT_DIR)
        assert len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".md")]) == 1
