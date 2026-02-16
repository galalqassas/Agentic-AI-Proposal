"""Chainlit UI for the Proposal Agent System.

Visualises the multi-agent workflow with correctly ordered, collapsible
steps nested under parent messages.  Only the final accepted proposal
appears as a standalone chat message.
"""

import os
from datetime import datetime, timezone

import chainlit as cl
from langchain_core.messages import HumanMessage

from data_layer import JsonDataLayer
from graph.graph import build_graph, QUALITY_THRESHOLD, MAX_ITERATIONS

# ── Initialisation ───────────────────────────────────────────────────
APP_GRAPH = build_graph()

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Nodes whose events we handle in the stream
MAJOR_NODES = frozenset({"planner", "researcher", "writer", "evaluator", "output"})

# Phase labels for the two parent messages
_PHASE1_LABEL = "📋 Planning & researching your proposal…"
_PHASE2_LABEL = "✍️ Drafting & refining your proposal…"

# Human-readable step names
_STEP_LABELS = {
    "planner": "Planner",
    "researcher": "Researcher",
    "output": "Output",
}


def _writer_label(n: int) -> str:
    return f"Writer (attempt {n})"


def _evaluator_label(n: int) -> str:
    return f"Evaluator (attempt {n})"


# ── Helpers ──────────────────────────────────────────────────────────
def _utc_now() -> str:
    """ISO-8601 UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def _score_emoji(score: float) -> str:
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
    rows = "\n".join(
        f"| {_score_emoji(v)} | {dim} | **{v}** |"
        for dim, v in dimension_scores.items()
    )
    emoji = _score_emoji(overall_score)
    return (
        f"## {emoji} Evaluation Scorecard\n\n"
        f"| | Dimension | Score |\n"
        f"|---|-----------|-------|\n"
        f"{rows}\n\n"
        f"**Overall: {overall_score}/10** · Revision {revision_count}/{MAX_ITERATIONS}"
    )


async def _end_step(step: cl.Step) -> None:
    """Mark a step as finished and push the update to the UI."""
    step.end = _utc_now()
    await step.update()


async def _make_step(name: str, parent_id: str) -> cl.Step:
    """Create, register, and send a step nested under *parent_id*."""
    step = cl.Step(name=name, type="run")
    step.parent_id = parent_id
    await step.send()
    return step


# ── Starters ─────────────────────────────────────────────────────────
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(label="📋 Grant Proposal",       message="Write a grant proposal for a non-profit focusing on renewable energy education."),
        cl.Starter(label="💼 Business Plan",         message="Create a business plan for a new AI startup targeting healthcare diagnostics."),
        cl.Starter(label="⚙️ Technical Proposal",    message="Draft a technical proposal for migrating a company's infrastructure to the cloud."),
        cl.Starter(label="💰 Sales Proposal",        message="Write a persuasive sales proposal for an enterprise SaaS analytics platform."),
        cl.Starter(label="📅 Project Proposal",      message="Create a project proposal for building a mobile app for smart city transportation."),
        cl.Starter(label="🔬 Research Proposal",     message="Draft a research proposal studying the impact of AI on financial markets."),
        cl.Starter(label="🤝 Partnership Proposal",  message="Write a strategic partnership proposal between a fintech startup and a major bank."),
        cl.Starter(label="📝 General Proposal",      message="Create a general proposal for launching a community mentorship programme."),
    ]


# ── Auth & data layer ────────────────────────────────────────────────

@cl.header_auth_callback
async def header_auth_callback(headers: dict) -> cl.User:
    """Auto-authenticate every visitor — no login page."""
    return cl.User(identifier="default", metadata={"role": "admin"})


@cl.data_layer
def get_data_layer():
    return JsonDataLayer()


# ── Chat lifecycle ───────────────────────────────────────────────────

@cl.on_chat_start
async def start():
    cl.user_session.set("graph", APP_GRAPH)
    cl.user_session.set("config", {"configurable": {"thread_id": cl.context.session.id}})
    cl.user_session.set("processed_ids", set())


@cl.on_chat_resume
async def on_chat_resume(thread: dict):
    """Restore session state when a user reopens a past conversation."""
    cl.user_session.set("graph", APP_GRAPH)
    cl.user_session.set("config", {"configurable": {"thread_id": thread["id"]}})
    cl.user_session.set("processed_ids", set())


@cl.on_message
async def main(message: cl.Message):
    """Stream graph execution with steps nested under parent messages."""
    graph = cl.user_session.get("graph")
    config = cl.user_session.get("config")

    state = graph.get_state(config)

    if state.next:
        await graph.aupdate_state(
            config, {"user_feedback": message.content}, as_node="researcher"
        )
        stream = graph.astream_events(None, config, version="v2")
    else:
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

    # ── Session-scoped tracking ──────────────────────────────────
    active_steps: dict[str, cl.Step] = {}
    root_msg: cl.Message | None = None   # current parent message
    attempt = 1                          # writer/evaluator attempt counter

    async for event in stream:
        kind = event["event"]
        name = event["name"]
        data = event["data"]

        # ── Node Start ───────────────────────────────────────────
        if kind == "on_chain_start" and name in MAJOR_NODES:
            run_id = event.get("run_id")
            processed = cl.user_session.get("processed_ids") or set()
            cl.user_session.set("processed_ids", processed)
            if run_id in processed:
                continue
            processed.add(run_id)

            # Phase 1 parent (planner / researcher)
            if name in ("planner", "researcher") and root_msg is None:
                root_msg = cl.Message(content=_PHASE1_LABEL)
                await root_msg.send()

            # Phase 2 parent (writer / evaluator loop)
            if name == "writer" and (root_msg is None or root_msg.content == _PHASE1_LABEL):
                root_msg = cl.Message(content=_PHASE2_LABEL)
                await root_msg.send()
                attempt = 1

            # Choose label
            if name == "writer":
                label = _writer_label(attempt)
            elif name == "evaluator":
                label = _evaluator_label(attempt)
            else:
                label = _STEP_LABELS.get(name, name.capitalize())

            step = await _make_step(label, root_msg.id)
            active_steps[name] = step

        # ── LLM Streaming ────────────────────────────────────────
        elif kind == "on_chat_model_stream":
            chunk_content = data["chunk"].content
            if not chunk_content:
                continue
            node = event.get("metadata", {}).get("langgraph_node")
            if node and node in active_steps:
                await active_steps[node].stream_token(chunk_content)

        # ── Node End ─────────────────────────────────────────────
        elif kind == "on_chain_end" and name in MAJOR_NODES:
            if name not in active_steps:
                continue

            step = active_steps.pop(name)
            output = data.get("output")

            if name == "planner":
                step.output = f"**Plan Generated**\n\n{output.get('plan', '')}"
                await _end_step(step)

            elif name == "researcher":
                await _finish_researcher(step, output)

            elif name == "writer":
                step.output = output.get("draft", "")
                await _end_step(step)

            elif name == "evaluator":
                await _finish_evaluator(step, output, root_msg)
                attempt += 1

            elif name == "output":
                await _finish_output(step, output)


# ── Node-end helpers ─────────────────────────────────────────────────
async def _finish_researcher(step: cl.Step, output: dict) -> None:
    """Finalise the researcher step, then send the HITL checkpoint."""
    research_data = output.get("research_data", "")
    search_queries = output.get("search_queries", [])

    step.output = f"**Research Complete**\n\n{research_data}"
    await _end_step(step)

    query_list = "\n".join(f"  {i}. {q}" for i, q in enumerate(search_queries, 1))
    query_section = (
        f"**🔎 Queries used ({len(search_queries)}):**\n{query_list}"
        if search_queries
        else ""
    )

    preview = research_data[:600].rstrip()
    if len(research_data) > 600:
        preview += "\n\n*(…see full brief in the step above)*"

    actions = [
        cl.Action(name="proceed",           value="proceed",    label="✅ Proceed to Write", payload={"value": "proceed"}),
        cl.Action(name="edit_requirements",  value="edit",       label="✏️ Edit Requirements", payload={"value": "edit"}),
        cl.Action(name="reresearch",         value="reresearch", label="🔄 Re-research",      payload={"value": "reresearch"}),
    ]

    await cl.Message(
        content=(
            "## ✨ Research Phase Complete\n\n"
            f"{query_section}\n\n"
            "---\n\n"
            f"### Research Preview\n{preview}\n\n"
            "---\n\n"
            "Review the findings above. You can **proceed**, **edit requirements**, "
            "or **re-run research** with different focus."
        ),
        actions=actions,
    ).send()


async def _finish_evaluator(
    step: cl.Step,
    output: dict,
    root_msg: cl.Message | None,
) -> None:
    """Render the scorecard inside the evaluator step."""
    score = output.get("score", 0.0)
    critique = output.get("critique", "")
    dimension_scores = output.get("dimension_scores", {})
    revision_count = output.get("revision_count", 1)

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

    step.output = scorecard
    await _end_step(step)

    # Update the parent message with a brief status
    if root_msg:
        status = (
            f"✅ Score **{score}/10** — proposal accepted!"
            if score >= QUALITY_THRESHOLD
            else f"🔄 Score **{score}/10** — revising (attempt {revision_count + 1})…"
        )
        root_msg.content = f"{_PHASE2_LABEL}\n\n{status}"
        await root_msg.update()


async def _finish_output(step: cl.Step, output: dict) -> None:
    """Send the final proposal as a standalone top-level message."""
    step.output = "Saved to disk."
    await _end_step(step)

    # Retrieve the draft from the graph state via the output node's return
    # (output node doesn't return draft, so read it from the step's graph state)
    # We access it via the session graph state instead.
    graph = cl.user_session.get("graph")
    config = cl.user_session.get("config")
    final_state = graph.get_state(config)
    draft = final_state.values.get("draft", "")

    if not draft:
        return

    elements = [
        cl.File(
            name="proposal_final.md",
            content=draft.encode("utf-8"),
            display="inline",
        )
    ]

    await cl.Message(
        content=f"# 📄 Final Proposal\n\n{draft}",
        elements=elements,
    ).send()


# ── Action callbacks ─────────────────────────────────────────────────
@cl.action_callback("proceed")
async def on_proceed(action: cl.Action):
    await action.remove()
    await main(cl.Message(content="Proceed", author="User"))


@cl.action_callback("edit_requirements")
async def on_edit(action: cl.Action):
    await action.remove()
    await cl.Message(
        content="✏️ **Please type your specific requirements below.**\n\nI'll incorporate them into the proposal draft."
    ).send()


@cl.action_callback("reresearch")
async def on_reresearch(action: cl.Action):
    await action.remove()
    await cl.Message(
        content="🔄 **Tell me what to focus the new research on.**\n\nFor example: *\"Focus more on competitor pricing\"* or *\"Research the European market instead\"*."
    ).send()
