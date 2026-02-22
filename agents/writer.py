"""Writer agent – generates a full proposal draft based on the plan and research."""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from graph.state import AgentState
from utils.llm import get_llm
from utils.templates import PROPOSAL_TEMPLATES


def writer_node(state: AgentState) -> dict:
    """LangGraph node – generates the proposal draft."""
    llm = get_llm(model="openai/gpt-oss-120b", temperature=0.4)

    plan = state.get("plan", "")
    research_data = state.get("research_data", "")
    proposal_type = state.get("proposal_type", "General")
    user_feedback = state.get("user_feedback", "")
    critique = state.get("critique", "")
    revision_count = state.get("revision_count", 0)

    # Select template (fallback to General if type not found)
    template_str = PROPOSAL_TEMPLATES.get(proposal_type, PROPOSAL_TEMPLATES["General"])

    # If there is user feedback, prepend it to the research context
    if user_feedback:
        research_data = (
            f"### Additional User Requirements\n{user_feedback}\n\n{research_data}"
        )

    # If revising, include the evaluator's critique
    if revision_count > 0 and critique:
        research_data = (
            f"### Evaluator Feedback (Revision {revision_count})\n"
            f"Address these issues in this revision:\n{critique}\n\n"
            f"{research_data}"
        )

    # We can use a simple template here since the instructions are embedded
    prompt = ChatPromptTemplate.from_template(template_str)

    messages = prompt.format_messages(plan=plan, research_data=research_data)
    response = llm.invoke(messages)

    content = response.content if hasattr(response, "content") else str(response)

    return {
        "messages": [AIMessage(content=f"Draft created ({len(content)} chars)")],
        "draft": content,
    }
