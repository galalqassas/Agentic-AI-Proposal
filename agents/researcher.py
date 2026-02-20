"""Research agent – gathers contextual data using the Tavily search API.

The researcher:
1. Reads the plan from the Planner (specifically the "Research Needed" items).
2. Uses Pydantic structured output to extract targeted search queries.
3. Calls Tavily for each query.
4. Synthesises the results into a structured research brief.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient

from agents.models import SearchQueries
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
You are a Senior Research Analyst. Given the proposal plan below, \
generate a set of highly specific, actionable search queries that will \
uncover data to personalise and strengthen the proposal.

**Focus areas (pick what's relevant):**
- Recipient / target company: recent news, leadership, strategy, pain points
- Industry: market size, growth rate, emerging trends, key statistics
- Competitors: pricing models, strengths, weaknesses, market share
- Regulations: relevant standards, compliance requirements, certifications
- Case studies: success stories in similar engagements

**Rules:**
- Generate at most 5 queries. Fewer is fine if the plan is narrow.
- Each query should be specific enough to return useful results on its own.
- Do NOT generate generic queries like "proposal writing tips"."""

SYNTHESIS_PROMPT = """\
You are a Research Analyst preparing a briefing document for a proposal \
writer. Synthesise the raw search results below into a clear, well-organised \
**Research Brief**.

**Guidelines:**
- Group findings by topic (e.g. "Company Background", "Market Trends").
- Lead with the most impactful data points — statistics, revenue figures, \
growth percentages, and direct quotes.
- Always cite sources with their URLs.
- Flag any conflicting data points and note recency of information.
- Keep the brief concise — focus on what directly strengthens the proposal."""

_query_prompt = ChatPromptTemplate.from_messages([
    ("system", QUERY_EXTRACTION_PROMPT),
    ("human", "### Plan\n{plan}\n\n### Additional User Feedback\n{user_feedback}"),
])

_synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", SYNTHESIS_PROMPT),
    ("human", "### Raw Results\n{raw_results}"),
])


# ── Helpers ──────────────────────────────────────────────────────────
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

    # Step 1: Extract queries using structured output
    structured_llm = llm.with_structured_output(SearchQueries)
    user_feedback = state.get("user_feedback", "")
    query_messages = _query_prompt.format_messages(
        plan=plan,
        user_feedback=user_feedback
    )
    query_result: SearchQueries = structured_llm.invoke(query_messages)
    queries = query_result.queries[:5]

    # Step 2: Search
    raw_results = _search_tavily(queries)

    # Step 3: Synthesise (free-form text is fine here)
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
