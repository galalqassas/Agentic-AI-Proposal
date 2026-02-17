"""Tests for the Evaluator agent."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage
from agents.evaluator import evaluator_node
from agents.models import EvaluationOutput

# ── Sample Pydantic output ───────────────────────────────────────────
SAMPLE_EVALUATION = EvaluationOutput(
    clarity=8.0,
    persuasiveness=7.0,
    completeness=9.0,
    structure=8.0,
    specificity=7.0,
    overall_score=8.5,
    critique=(
        "1. Make the executive summary more punchy.\n"
        "2. Add more specific data points about the industry.\n"
        "3. Clarify the timeline in section 4."
    ),
)

SAMPLE_PASSING_EVALUATION = EvaluationOutput(
    clarity=9.0,
    persuasiveness=9.0,
    completeness=9.0,
    structure=9.0,
    specificity=9.0,
    overall_score=9.0,
    critique="",
)


class TestEvaluatorNode:
    """Tests for the evaluator_node."""

    @patch("agents.evaluator.get_llm")
    def test_parses_score_and_critique(self, mock_get_llm):
        """Should extract score and critique from structured output."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = SAMPLE_EVALUATION
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [],
            "task": "Test task",
            "draft": "Test draft",
            "score": 0.0,
            "revision_count": 0,
        }

        result = evaluator_node(state)

        assert result["score"] == 8.5
        assert "Make the executive summary more punchy" in result["critique"]
        assert result["revision_count"] == 1

        # Verify with_structured_output was used
        mock_llm.with_structured_output.assert_called_once_with(EvaluationOutput)
        mock_structured.invoke.assert_called_once()

    @patch("agents.evaluator.get_llm")
    def test_dimension_scores_extracted(self, mock_get_llm):
        """Should return all 5 dimension scores."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = SAMPLE_EVALUATION
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [],
            "task": "Test",
            "draft": "Draft",
            "score": 0.0,
            "revision_count": 0,
        }

        result = evaluator_node(state)

        assert result["dimension_scores"]["Clarity"] == 8.0
        assert result["dimension_scores"]["Persuasiveness"] == 7.0
        assert result["dimension_scores"]["Completeness"] == 9.0
        assert result["dimension_scores"]["Structure"] == 8.0
        assert result["dimension_scores"]["Specificity"] == 7.0

    @patch("agents.evaluator.get_llm")
    def test_passing_score_no_critique(self, mock_get_llm):
        """High-scoring proposals should have empty critique."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = SAMPLE_PASSING_EVALUATION
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [],
            "task": "Test",
            "draft": "Great draft",
            "score": 0.0,
            "revision_count": 0,
        }

        result = evaluator_node(state)

        assert result["score"] == 9.0
        assert result["critique"] == ""

    @patch("agents.evaluator.get_llm")
    def test_increments_revision_count(self, mock_get_llm):
        """Should increment revision count from current state."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = SAMPLE_EVALUATION
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [],
            "task": "Test",
            "draft": "Draft",
            "score": 0.0,
            "revision_count": 2,
        }

        result = evaluator_node(state)

        assert result["revision_count"] == 3
