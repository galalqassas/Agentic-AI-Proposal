"""Tests for the Evaluator agent."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage
from agents.evaluator import evaluator_node, _extract_overall_score

SAMPLE_EVALUATION = """\
### Scores
Clarity: 8
Persuasiveness: 7
Completeness: 9
Structure: 8
Specificity: 7

### Overall Score
7.8

### Critique
1. Make the executive summary more punchy.
2. Add more specific data points about the industry.
3. Clarify the timeline in section 4.
"""

class TestEvaluatorNode:
    """Tests for the evaluator_node."""

    @patch("agents.evaluator.get_llm")
    def test_parses_score_and_critique(self, mock_get_llm):
        """Should extract score and critique from LLM output."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content=SAMPLE_EVALUATION)
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [],
            "task": "Test task",
            "draft": "Test draft",
            "score": 0.0,
            "revision_count": 0,
        }

        result = evaluator_node(state)

        assert result["score"] == 7.8
        assert "Make the executive summary more punchy" in result["critique"]
        assert result["revision_count"] == 1
        
        # Verify LLM call
        mock_llm.invoke.assert_called_once()

    def test_extract_score_regex(self):
        """Test regex robustness."""
        assert _extract_overall_score("### Overall Score\n8.5") == 8.5
        assert _extract_overall_score("### Overall Score: 9") == 9.0
        assert _extract_overall_score("No score here") == 0.0
