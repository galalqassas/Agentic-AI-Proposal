"""Tests for the Research agent.

These tests mock both the LLM and Tavily so they run offline.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage

from agents.researcher import (
    researcher_node,
    _extract_queries,
    _search_tavily,
)

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

SAMPLE_QUERIES_LLM_OUTPUT = """\
1. Acme Corp company overview and recent news
2. AI consulting market size 2026
3. Top AI consulting firms pricing comparison
"""

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
class TestExtractQueries:
    """Tests for the _extract_queries helper."""

    def test_parses_numbered_list(self):
        queries = _extract_queries(SAMPLE_QUERIES_LLM_OUTPUT)
        assert len(queries) == 3
        assert "Acme Corp company overview" in queries[0]

    def test_caps_at_five(self):
        text = "\n".join(f"{i}. query {i}" for i in range(1, 10))
        queries = _extract_queries(text)
        assert len(queries) <= 5

    def test_handles_empty_input(self):
        assert _extract_queries("") == []

    def test_handles_dash_prefix(self):
        text = "1- First query\n2- Second query"
        queries = _extract_queries(text)
        assert len(queries) == 2

    def test_handles_paren_prefix(self):
        text = "1) First query\n2) Second query"
        queries = _extract_queries(text)
        assert len(queries) == 2


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
        # LLM is called twice: query extraction, then synthesis
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            AIMessage(content=SAMPLE_QUERIES_LLM_OUTPUT),
            AIMessage(content=SAMPLE_SYNTHESIS),
        ]
        mock_get_llm.return_value = mock_llm

        mock_search.return_value = "Raw search results here"

        state = {
            "messages": [],
            "task": "Write a proposal for Acme Corp",
            "proposal_type": "Business",
            "plan": SAMPLE_PLAN,
            "research_data": "",
            "draft": "",
            "critique": "",
            "score": 0.0,
            "revision_count": 0,
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
            "draft": "",
            "critique": "",
            "score": 0.0,
            "revision_count": 0,
        }

        result = researcher_node(state)

        assert result["research_data"] == ""
        assert "No plan provided" in result["messages"][0].content

    @patch("agents.researcher._search_tavily")
    @patch("agents.researcher.get_llm")
    def test_calls_tavily_with_extracted_queries(self, mock_get_llm, mock_search):
        """Verify Tavily is called after query extraction."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            AIMessage(content="1. test query one\n2. test query two"),
            AIMessage(content="Synthesised brief"),
        ]
        mock_get_llm.return_value = mock_llm
        mock_search.return_value = "Raw results"

        state = {
            "messages": [],
            "task": "Test",
            "proposal_type": "Business",
            "plan": "### Research Needed\n- topic one\n- topic two",
            "research_data": "",
            "draft": "",
            "critique": "",
            "score": 0.0,
            "revision_count": 0,
        }

        researcher_node(state)

        mock_search.assert_called_once()
        # Should pass extracted queries
        queries_arg = mock_search.call_args[0][0]
        assert len(queries_arg) == 2
