"""Planner agent – analyses the user's proposal request and produces a
structured plan.

The planner:
1. Identifies the proposal type (Grant, Business, Technical, etc.).
2. Extracts key information already provided.
3. Lists missing information that should be researched.
4. Outputs a concise, structured plan with sections for the proposal.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from graph.state import AgentState
from utils.llm import get_llm

PLANNER_SYSTEM_PROMPT = """\
You are a Lead Proposal Strategist. Your job is to analyse a proposal \
request and produce a high-quality, concise plan.

**Instructions**
1. Identify the *proposal type* (one of: Grant, Business, Technical, \
Sales, Project, Research, Partnership – or "General" if unclear).
2. List what the user has already told you (key facts).
3. List what is *missing* and needs research (recipient background, \
industry data, competitor benchmarks, etc.).
4. Produce a **Proposal Plan** with numbered sections (e.g. Executive \
Summary, Problem Statement, Solution, Budget, Timeline, …).  \
Keep section names appropriate for the detected type.

**Output format** (use exactly these headings):
### Proposal Type
<type>

### Key Facts Provided
- …

### Research Needed
- …

### Proposal Plan
1. …
2. …
…
"""

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PLANNER_SYSTEM_PROMPT),
        ("human", "{task}"),
    ]
)


def planner_node(state: AgentState) -> dict:
    """LangGraph node – runs the planner and returns updated state."""
    llm = get_llm(model="openai/gpt-oss-120b", temperature=0.3)

    # Render the prompt, then invoke the LLM directly
    messages = _prompt.format_messages(task=state["task"])
    response = llm.invoke(messages)

    content = response.content if hasattr(response, "content") else str(response)
    proposal_type = _extract_proposal_type(content)

    return {
        "messages": [AIMessage(content=content)],
        "plan": content,
        "proposal_type": proposal_type,
    }


def _extract_proposal_type(plan_text: str) -> str:
    """Pull the proposal type from the structured output using regex."""
    import re
    # Robust extraction of the proposal type (handles "Type: Business" and "Type\nBusiness")
    if match := re.search(r"### Proposal Type[:\s]*([^\n#]+)", plan_text, re.IGNORECASE):
        return match.group(1).strip()
    return "General"
