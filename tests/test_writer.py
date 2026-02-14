"""Tests for the Writer agent."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage
from agents.writer import writer_node

class TestWriterNode:
    """Tests for the writer_node."""

    @patch("agents.writer.get_llm")
    def test_generates_draft(self, mock_get_llm):
        """Should generate a draft using the prompt template."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Generated Draft Content")
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [],
            "task": "Test task",
            "proposal_type": "Business",
            "plan": "Test Plan",
            "research_data": "Test Research",
            "draft": "",
            "critique": "",
            "score": 0.0,
            "revision_count": 0,
        }

        result = writer_node(state)

        assert result["draft"] == "Generated Draft Content"
        assert "messages" in result
        
        # Verify LLM call
        mock_llm.invoke.assert_called_once()
        # The prompt should contain the plan and research
        # We can inspect the calls if needed, but the main thing is it ran.

    @patch("agents.writer.get_llm")
    def test_handles_unknown_type(self, mock_get_llm):
        """Should fallback to General template for unknown types."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Draft")
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [],
            "task": "Test",
            "proposal_type": "UnknownType",
            "plan": "Plan",
            "research_data": "Research",
            "draft": "",
            "critique": "",
            "score": 0.0,
            "revision_count": 0,
        }

        writer_node(state)
        # Should complete successfully (using fallback template internal logic)
        mock_llm.invoke.assert_called_once()
