"""Chainlit UI for the Proposal Agent System.

This app visualizes the multi-agent workflow using Chainlit's Step API.
Each agent (Planner, Researcher, Writer, Evaluator) helps build the final proposal.
"""

import os

import chainlit as cl
from langchain_core.messages import HumanMessage

from graph.graph import build_graph, QUALITY_THRESHOLD, MAX_ITERATIONS

# ── Initialisation ───────────────────────────────────────────────────
APP_GRAPH = build_graph()

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Agent nodes we track in the event stream
MAJOR_NODES = frozenset({"planner", "researcher", "writer", "evaluator", "output"})


# ── Helpers ──────────────────────────────────────────────────────────
def _score_emoji(score: float) -> str:
    """Return a colour-coded emoji for a dimension score."""
    if score >= 9.0:
        return "🟢"
    if score >= 7.0:
        return "🟡"
    return "🔴"


def _build_scorecard(
    dimension_scores: dict,
    overall_score: float,
    revision_count: int,
) -> str:
    """Build a rich Markdown scorecard for the evaluator output."""
    rows = "\n".join(
        f"| {_score_emoji(v)} | {dim} | **{v}** |"
        for dim, v in dimension_scores.items()
    )
    overall_emoji = _score_emoji(overall_score)

    return (
        f"## {overall_emoji} Evaluation Scorecard\n\n"
        f"| | Dimension | Score |\n"
        f"|---|-----------|-------|\n"
        f"{rows}\n\n"
        f"**Overall: {overall_score}/10** · Revision {revision_count}/{MAX_ITERATIONS}"
    )


# ── Starters ─────────────────────────────────────────────────────────
@cl.set_starters
async def set_starters():
    """Provide starter prompts for all supported proposal types."""
    return [
        cl.Starter(
            label="📋 Grant Proposal",
            message="Write a grant proposal for a non-profit focusing on renewable energy education.",
        ),
        cl.Starter(
            label="💼 Business Plan",
            message="Create a business plan for a new AI startup targeting healthcare diagnostics.",
        ),
        cl.Starter(
            label="⚙️ Technical Proposal",
            message="Draft a technical proposal for migrating a company's infrastructure to the cloud.",
        ),
        cl.Starter(
            label="💰 Sales Proposal",
            message="Write a persuasive sales proposal for an enterprise SaaS analytics platform.",
        ),
        cl.Starter(
            label="📅 Project Proposal",
            message="Create a project proposal for building a mobile app for smart city transportation.",
        ),
        cl.Starter(
            label="🔬 Research Proposal",
            message="Draft a research proposal studying the impact of AI on financial markets.",
        ),
        cl.Starter(
            label="🤝 Partnership Proposal",
            message="Write a strategic partnership proposal between a fintech startup and a major bank.",
        ),
        cl.Starter(
            label="📝 General Proposal",
            message="Create a general proposal for launching a community mentorship programme.",
        ),
    ]


# ── Chat lifecycle ───────────────────────────────────────────────────
@cl.on_chat_start
async def start():
    """Initialize the chat session."""
    cl.user_session.set("graph", APP_GRAPH)
    cl.user_session.set("config", {"configurable": {"thread_id": cl.context.session.id}})
    cl.user_session.set("processed_ids", set())

    await cl.Message(
        content=(
            "# 🚀 Welcome to the Proposal Agent!\n\n"
            "I'm your AI specialist for crafting **research-backed, high-quality proposals**.\n\n"
            "### My Process\n"
            "1. **Plan** 📝 — I outline the structure\n"
            "2. **Research** 🔍 — I gather real-world data\n"
            "3. **Write** ✍️ — I draft the full proposal\n"
            "4. **Evaluate** 🧐 — I score and refine until excellence (target: 9.5/10)\n\n"
            "**Pick a starter below or describe your proposal to begin!**"
        ),
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle user input, handle interrupts, and stream the graph execution."""
    graph = cl.user_session.get("graph")
    config = cl.user_session.get("config")

    # Check current state to see if we are resuming from an interrupt
    state = graph.get_state(config)

    if state.next:
        # Resuming after the researcher interrupt — user message is the feedback
        await graph.aupdate_state(
            config, {"user_feedback": message.content}, as_node="researcher"
        )
        stream = graph.astream_events(None, config, version="v2")
    else:
        # Start fresh
        inputs = {
            "task": message.content,
            "messages": [HumanMessage(content=message.content)],
            "revision_count": 0,
            "score": 0.0,
            "user_feedback": "",
            "search_queries": [],
            "dimension_scores": {},
        }
        stream = graph.astream_events(inputs, config, version="v2")

    # Track active steps and main response
    active_steps: dict[str, cl.Step] = {}
    main_response: cl.Message | None = None

    async for event in stream:
        kind = event["event"]
        name = event["name"]
        data = event["data"]

        # ── Node Start ───────────────────────────────────────────
        if kind == "on_chain_start" and name in MAJOR_NODES:
            run_id = event.get("run_id")
            processed_ids = cl.user_session.get("processed_ids") or set()
            cl.user_session.set("processed_ids", processed_ids)

            if run_id in processed_ids:
                continue
            processed_ids.add(run_id)

            step = cl.Step(name=name.capitalize(), type="run")
            await step.send()
            active_steps[name] = step

            # Prepare the main response message for the writer's draft
            if name == "writer":
                main_response = cl.Message(content="")
                await main_response.send()

        # ── LLM Streaming ────────────────────────────────────────
        elif kind == "on_chat_model_stream":
            content = data["chunk"].content
            if not content:
                continue

            # Route tokens deterministically via LangGraph metadata
            node = event.get("metadata", {}).get("langgraph_node")
            if not node or node not in active_steps:
                continue

            # Writer tokens go to the main response only (not the step)
            if node == "writer" and main_response:
                await main_response.stream_token(content)
            else:
                await active_steps[node].stream_token(content)

        # ── Node End ─────────────────────────────────────────────
        elif kind == "on_chain_end" and name in MAJOR_NODES:
            if name not in active_steps:
                continue

            step = active_steps.pop(name)
            output = data.get("output")

            if name == "planner":
                step.output = f"**Plan Generated**\n\n{output.get('plan', '')}"
                await step.update()

            elif name == "researcher":
                await _handle_researcher_end(step, output)

            elif name == "writer":
                await _handle_writer_end(step, output, main_response)

            elif name == "evaluator":
                await _handle_evaluator_end(step, output)

            elif name == "output":
                step.output = "Process complete."
                await step.update()


# ── Node-end handlers ────────────────────────────────────────────────
async def _handle_researcher_end(step: cl.Step, output: dict) -> None:
    """Display research results and present the human-in-the-loop checkpoint."""
    research_data = output.get("research_data", "")
    search_queries = output.get("search_queries", [])

    step.output = f"**Research Complete**\n\n{research_data}"
    await step.update()

    # Build a summary of what was searched
    query_list = "\n".join(f"  {i}. {q}" for i, q in enumerate(search_queries, 1))
    query_section = f"**🔎 Queries used ({len(search_queries)}):**\n{query_list}" if search_queries else ""

    # Truncate the research brief for the checkpoint message
    preview = research_data[:600].rstrip()
    if len(research_data) > 600:
        preview += "\n\n*(…see full brief in the step above)*"

    actions = [
        cl.Action(name="proceed", value="proceed", label="✅ Proceed to Write", payload={"value": "proceed"}),
        cl.Action(name="edit_requirements", value="edit", label="✏️ Edit Requirements", payload={"value": "edit"}),
        cl.Action(name="reresearch", value="reresearch", label="🔄 Re-research", payload={"value": "reresearch"}),
    ]

    await cl.Message(
        content=(
            "## ✨ Research Phase Complete\n\n"
            f"{query_section}\n\n"
            "---\n\n"
            f"### Research Preview\n{preview}\n\n"
            "---\n\n"
            "Review the findings above. You can **proceed**, **edit requirements**, or **re-run research** with different focus."
        ),
        actions=actions,
    ).send()


async def _handle_writer_end(
    step: cl.Step,
    output: dict,
    main_response: cl.Message | None,
) -> None:
    """Finalize the writer step and provide a downloadable draft file."""
    # 1. Close the step first
    step.output = "Draft generated."
    await step.update()

    # 2. Finalize the streamed main response
    if main_response:
        await main_response.update()

    # 3. Offer the draft as a downloadable file
    draft_content = output.get("draft", "")
    elements = [
        cl.File(
            name="proposal_draft.md",
            content=draft_content.encode("utf-8"),
            display="inline",
        )
    ]
    await cl.Message(content="📄 **Here is your proposal draft:**", elements=elements).send()


async def _handle_evaluator_end(step: cl.Step, output: dict) -> None:
    """Display the rich scorecard and revision status."""
    score = output.get("score", 0.0)
    critique = output.get("critique", "")
    dimension_scores = output.get("dimension_scores", {})
    revision_count = output.get("revision_count", 1)

    # Update the step with the raw evaluation
    step.output = f"**Score: {score}/10**\n\n{critique}"
    await step.update()

    # Build and send the rich scorecard message
    scorecard = _build_scorecard(dimension_scores, score, revision_count)

    if critique and score < QUALITY_THRESHOLD:
        scorecard += f"\n\n### 📝 Critique\n{critique}"

    if score < QUALITY_THRESHOLD and revision_count < MAX_ITERATIONS:
        scorecard += (
            f"\n\n---\n⚡ **Score {score}/10** is below the {QUALITY_THRESHOLD} threshold. "
            f"Revising draft (attempt {revision_count + 1}/{MAX_ITERATIONS})…"
        )
    elif score >= QUALITY_THRESHOLD:
        scorecard += "\n\n---\n🎉 **Excellent!** The proposal meets the quality bar."

    await cl.Message(content=scorecard).send()


# ── Action callbacks ─────────────────────────────────────────────────
@cl.action_callback("proceed")
async def on_proceed(action: cl.Action):
    """User approves the research — continue to the writer."""
    await action.remove()
    await main(cl.Message(content="Proceed", author="User"))


@cl.action_callback("edit_requirements")
async def on_edit(action: cl.Action):
    """User wants to add specific requirements before writing."""
    await action.remove()
    await cl.Message(
        content="✏️ **Please type your specific requirements below.**\n\nI'll incorporate them into the proposal draft."
    ).send()


@cl.action_callback("reresearch")
async def on_reresearch(action: cl.Action):
    """User wants the researcher to gather fresh data."""
    await action.remove()
    await cl.Message(
        content="🔄 **Tell me what to focus the new research on.**\n\nFor example: *\"Focus more on competitor pricing\"* or *\"Research the European market instead\"*."
    ).send()
