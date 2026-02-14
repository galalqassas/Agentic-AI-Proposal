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

# ── Mock responses ───────────────────────────────────────────────────
MOCK_PLAN = """### Proposal Type\nBusiness\n### Plan\n..."""
MOCK_QUERIES = "1. query"
MOCK_BRIEF = "Research brief"
MOCK_DRAFT_V1 = "Draft V1"
MOCK_EVAL_LOW = """### Overall Score\n5.0\n### Critique\nImprove content."""
MOCK_DRAFT_V2 = "Draft V2"
MOCK_EVAL_HIGH = """### Overall Score\n9.0\n### Critique\nGood job."""

class TestFullPipeline:
    """Tests the complete graph execution."""

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
        """Should loop once (writer -> evaluator -> writer -> evaluator -> output)."""
        
        # 1. Planner
        mock_planner_llm.return_value.invoke.return_value = AIMessage(content=MOCK_PLAN)
        
        # 2. Researcher (Queries + Synthesis)
        mock_researcher_llm.return_value.invoke.side_effect = [
            AIMessage(content=MOCK_QUERIES),
            AIMessage(content=MOCK_BRIEF),
        ]
        mock_tavily.return_value = "Raw results"
        
        # 3. Writer (called twice: initial + refinement)
        mock_writer_llm.return_value.invoke.side_effect = [
            AIMessage(content=MOCK_DRAFT_V1),
            AIMessage(content=MOCK_DRAFT_V2),
        ]
        
        # 4. Evaluator (called twice: low score -> high score)
        mock_evaluator_llm.return_value.invoke.side_effect = [
            AIMessage(content=MOCK_EVAL_LOW),   # Score 5.0 -> Loop
            AIMessage(content=MOCK_EVAL_HIGH),  # Score 9.0 -> Exit
        ]

        # Run graph
        app = build_graph()
        initial_state = {
            "messages": [],
            "task": "Write a proposal",
            "proposal_type": "",
            "plan": "",
            "research_data": "",
            "draft": "",
            "critique": "",
            "score": 0.0,
            "revision_count": 0,
        }
        
        result = app.invoke(initial_state)
        
        # Assertions
        assert result["score"] == 9.0
        assert result["revision_count"] == 2  # 0->1 (low) -> 2 (high)
        assert result["draft"] == MOCK_DRAFT_V2
        
        # Verify loop execution
        assert mock_writer_llm.return_value.invoke.call_count == 2
        assert mock_evaluator_llm.return_value.invoke.call_count == 2
        
        # Verify output creation
        assert os.path.exists(OUTPUT_DIR)
        assert len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".md")]) == 1
