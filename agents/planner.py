"""Planner agent – analyses the user's proposal request and produces a
structured plan with Pydantic-validated output.

The planner:
1. Identifies the proposal type (Grant, Business, Technical, etc.).
2. Extracts key information already provided.
3. Lists missing information that should be researched.
4. Outputs a concise, structured plan with sections for the proposal.
5. Asks the user *only* for information that cannot be found online.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from agents.models import PlannerOutput
from graph.state import AgentState
from utils.llm import get_llm

PLANNER_SYSTEM_PROMPT = """\
You are a Lead Proposal Strategist with 20+ years of experience crafting \
winning proposals across industries. Your role is to analyse a proposal \
request and produce a clear, actionable plan.

## Your Responsibilities

1. **Classify** the proposal type as one of: Grant, Business, Technical, \
Sales, Project, Research, Partnership, or General.

2. **Extract key facts** the user has already shared (names, dates, goals, \
constraints, etc.).

3. **Identify research gaps** — topics a web search can fill (recipient \
background, industry data, competitor benchmarks, regulations).

4. **Plan the proposal** — list the exact section titles in the order they \
should appear. Tailor section names to the detected proposal type \
(e.g. a Grant proposal needs "Impact & Sustainability" while a Sales \
proposal needs "ROI Analysis").

5. **Ask the user ONLY what you cannot find online.** Examples of \
unsearchable information:
   - Which company / organisation is *sending* this proposal?
   - Internal budget constraints or pricing
   - Confidential project timelines or deadlines
   - Proprietary capabilities or past performance details

## Rules for Questions
- Ask as few questions as possible. If the user's message already \
provides enough context, return an empty list.
- Each question must be short, specific, and easy to answer \
(ideally one sentence or a number with an example).
- Never ask about things you can research online (company info, \
market data, industry trends).
"""

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PLANNER_SYSTEM_PROMPT),
        ("human", "{task}"),
    ]
)


def planner_node(state: AgentState) -> dict:
    """LangGraph node – runs the planner and returns structured state."""
    llm = get_llm(model="openai/gpt-oss-120b", temperature=0.3)
    structured_llm = llm.with_structured_output(PlannerOutput)

    messages = _prompt.format_messages(task=state["task"])
    result: PlannerOutput = structured_llm.invoke(messages)

    # Build a human-readable plan string for downstream agents
    plan_text = _format_plan(result)

    return {
        "messages": [AIMessage(content=plan_text)],
        "plan": plan_text,
        "proposal_type": result.proposal_type,
        "questions_for_user": result.questions_for_user,
    }


def _format_plan(output: PlannerOutput) -> str:
    """Convert structured output into a readable Markdown plan."""
    facts = "\n".join(f"- {f}" for f in output.key_facts) or "- (none provided)"
    research = "\n".join(f"- {r}" for r in output.research_needed) or "- (none needed)"
    sections = "\n".join(f"{i}. {s}" for i, s in enumerate(output.proposal_sections, 1))

    return (
        f"### Proposal Type\n{output.proposal_type}\n\n"
        f"### Key Facts Provided\n{facts}\n\n"
        f"### Research Needed\n{research}\n\n"
        f"### Proposal Plan\n{sections}"
    )
