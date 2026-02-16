"""Research agent – gathers contextual data using the Tavily search API.

The researcher:
1. Reads the plan from the Planner (specifically the "Research Needed" items).
2. Formulates targeted search queries.
3. Calls Tavily for each query.
4. Synthesises the results into a structured research brief.
"""

from __future__ import annotations

import os
import re
from dotenv import load_dotenv

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient

from graph.state import AgentState
from utils.llm import get_llm

load_dotenv()

# ── Tavily client ───────────────────────────────────────────────────
_tavily_api_key = os.getenv("TAVILY_API_KEY", "")


def _get_tavily_client() -> TavilyClient:
    """Return a Tavily client (lazy so tests can mock the key)."""
    key = os.getenv("TAVILY_API_KEY", _tavily_api_key)
    if not key:
        raise ValueError("TAVILY_API_KEY is not set in the environment.")
    return TavilyClient(api_key=key)


# ── Prompts ─────────────────────────────────────────────────────────
QUERY_EXTRACTION_PROMPT = """\
You are a research assistant. Given the proposal plan below, extract \
a list of **specific search queries** that will help personalise and \
strengthen the proposal.

Focus on:
- Recipient / company background
- Industry trends and statistics
- Competitor benchmarks
- Relevant regulations or standards

Return ONLY a numbered list of search queries (max 5). No extra text."""

SYNTHESIS_PROMPT = """\
You are a research analyst. Summarise the following raw search results \
into a concise **Research Brief** that a proposal writer can reference.

Organise by topic. Include specific data points, statistics, and quotes \
where available. Cite sources with URLs."""

_query_prompt = ChatPromptTemplate.from_messages([
    ("system", QUERY_EXTRACTION_PROMPT),
    ("human", "### Plan\n{plan}"),
])

_synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", SYNTHESIS_PROMPT),
    ("human", "### Raw Results\n{raw_results}"),
])


# ── Helpers ──────────────────────────────────────────────────────────
def _extract_queries(llm_output: str) -> list[str]:
    """Extract up to 5 numbered queries using regex."""
    queries = re.findall(r"^\s*\d+[.\)\-]\s*(.+)$", llm_output, re.MULTILINE)
    return queries[:5]


def _search_tavily(queries: list[str]) -> str:
    """Run queries through Tavily and return concatenated results."""
    client = _get_tavily_client()
    results: list[str] = []

    for q in queries:
        try:
            res = client.search(query=q, max_results=3).get("results", [])
            results.extend(
                f"**{r.get('title')}**\n{r.get('content')}\nSource: {r.get('url')}"
                for r in res
            )
        except Exception as exc:
            results.append(f"[Search failed for '{q}': {exc}]")

    return "\n---\n".join(results) if results else "No results found."


# ── Graph node ───────────────────────────────────────────────────────
def researcher_node(state: AgentState) -> dict:
    """LangGraph node – researches the plan and returns a research brief."""
    llm = get_llm(model="openai/gpt-oss-120b", temperature=1)
    plan = state.get("plan", "")

    if not plan:
        return {
            "messages": [AIMessage(content="No plan provided to research.")],
            "research_data": "",
            "search_queries": [],
        }

    # Step 1: Extract queries from the plan
    query_messages = _query_prompt.format_messages(plan=plan)
    raw_queries = llm.invoke(query_messages)
    query_text = (
        raw_queries.content
        if hasattr(raw_queries, "content")
        else str(raw_queries)
    )
    queries = _extract_queries(query_text)

    # Step 2: Search
    raw_results = _search_tavily(queries)

    # Step 3: Synthesise
    synthesis_messages = _synthesis_prompt.format_messages(raw_results=raw_results)
    synthesis = llm.invoke(synthesis_messages)
    brief = (
        synthesis.content
        if hasattr(synthesis, "content")
        else str(synthesis)
    )

    return {
        "messages": [AIMessage(content=brief)],
        "research_data": brief,
        "search_queries": queries,
    }
