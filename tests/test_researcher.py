"""Tests for the Research agent.

These tests mock both the LLM and Tavily so they run offline.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage

from agents.researcher import (
    researcher_node,
    _search_tavily,
)
from agents.models import SearchQueries

# ── Sample data ──────────────────────────────────────────────────────
SAMPLE_PLAN = """\
### Proposal Type
Business

### Research Needed
- Acme Corp company background
- AI consulting market trends 2026
- Competitor pricing benchmarks

### Proposal Plan
1. Executive Summary
2. Problem Statement
3. Proposed Solution
"""

SAMPLE_QUERIES = SearchQueries(
    queries=[
        "Acme Corp company overview and recent news",
        "AI consulting market size 2026",
        "Top AI consulting firms pricing comparison",
    ]
)

SAMPLE_TAVILY_RESPONSE = {
    "results": [
        {
            "title": "Acme Corp Profile",
            "url": "https://example.com/acme",
            "content": "Acme Corp is a Fortune 500 company...",
        },
        {
            "title": "AI Consulting Market Report",
            "url": "https://example.com/ai-market",
            "content": "The AI consulting market is projected to reach $50B...",
        },
    ]
}

SAMPLE_SYNTHESIS = """\
## Research Brief

### Acme Corp Background
Acme Corp is a Fortune 500 company specialising in manufacturing.
Source: https://example.com/acme

### Market Trends
The AI consulting market is projected to reach $50B by 2027.
Source: https://example.com/ai-market
"""


# ── Unit tests ───────────────────────────────────────────────────────
class TestSearchTavily:
    """Tests for the _search_tavily helper (mocked)."""

    @patch("agents.researcher._get_tavily_client")
    def test_returns_formatted_results(self, mock_client_fn):
        mock_client = MagicMock()
        mock_client.search.return_value = SAMPLE_TAVILY_RESPONSE
        mock_client_fn.return_value = mock_client

        results = _search_tavily(["test query"])

        assert "Acme Corp Profile" in results
        assert "https://example.com/acme" in results
        mock_client.search.assert_called_once()

    @patch("agents.researcher._get_tavily_client")
    def test_handles_search_failure(self, mock_client_fn):
        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("API error")
        mock_client_fn.return_value = mock_client

        results = _search_tavily(["failing query"])

        assert "Search failed" in results

    @patch("agents.researcher._get_tavily_client")
    def test_handles_empty_results(self, mock_client_fn):
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}
        mock_client_fn.return_value = mock_client

        results = _search_tavily(["empty query"])
        assert isinstance(results, str)


class TestResearcherNode:
    """Tests for the researcher_node graph function."""

    @patch("agents.researcher._search_tavily")
    @patch("agents.researcher.get_llm")
    def test_returns_research_data(self, mock_get_llm, mock_search):
        """researcher_node should return research_data in state."""
        # LLM is called twice: structured (queries) + free-form (synthesis)
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = SAMPLE_QUERIES
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm.invoke.return_value = AIMessage(content=SAMPLE_SYNTHESIS)
        mock_get_llm.return_value = mock_llm

        mock_search.return_value = "Raw search results here"

        state = {
            "messages": [],
            "task": "Write a proposal for Acme Corp",
            "proposal_type": "Business",
            "plan": SAMPLE_PLAN,
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

        result = researcher_node(state)

        assert "research_data" in result
        assert len(result["research_data"]) > 0
        assert "messages" in result
        assert len(result["messages"]) == 1

    @patch("agents.researcher.get_llm")
    def test_handles_empty_plan(self, mock_get_llm):
        """researcher_node should handle missing plan gracefully."""
        state = {
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
            "user_feedback": "",
            "questions_for_user": [],
        }

        result = researcher_node(state)

        assert result["research_data"] == ""
        assert "No plan provided" in result["messages"][0].content

    @patch("agents.researcher._search_tavily")
    @patch("agents.researcher.get_llm")
    def test_calls_tavily_with_extracted_queries(self, mock_get_llm, mock_search):
        """Verify Tavily is called after structured query extraction."""
        queries = SearchQueries(queries=["test query one", "test query two"])

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = queries
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm.invoke.return_value = AIMessage(content="Synthesised brief")
        mock_get_llm.return_value = mock_llm
        mock_search.return_value = "Raw results"

        state = {
            "messages": [],
            "task": "Test",
            "proposal_type": "Business",
            "plan": "### Research Needed\n- topic one\n- topic two",
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

        researcher_node(state)

        mock_search.assert_called_once()
        # Should pass extracted queries
        queries_arg = mock_search.call_args[0][0]
        assert len(queries_arg) == 2

    @patch("agents.researcher._search_tavily")
    @patch("agents.researcher.get_llm")
    def test_uses_structured_output_for_queries(self, mock_get_llm, mock_search):
        """Verify with_structured_output is called with SearchQueries."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = SearchQueries(queries=["q1"])
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm.invoke.return_value = AIMessage(content="Brief")
        mock_get_llm.return_value = mock_llm
        mock_search.return_value = "Results"

        state = {
            "messages": [],
            "task": "Test",
            "proposal_type": "Business",
            "plan": "### Research Needed\n- topic",
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

        researcher_node(state)

        mock_llm.with_structured_output.assert_called_once_with(SearchQueries)
