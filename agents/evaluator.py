"""Evaluator agent – scores the proposal draft and provides critique.

Uses Pydantic structured output to return validated scores and feedback.

The evaluator:
1. Reads the draft and the original task.
2. Scores it on 5 dimensions (0-10).
3. Calculates an average score.
4. Generates a critique if the score is below threshold.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from agents.models import EvaluationOutput
from graph.state import AgentState
from utils.llm import get_llm

# ── Constants ────────────────────────────────────────────────────────
SCORE_DIMENSIONS = ("Clarity", "Persuasiveness", "Completeness", "Structure", "Specificity")

EVALUATOR_SYSTEM_PROMPT = """\
You are an expert Proposal Evaluator with deep experience reviewing \
proposals across industries. Your evaluation must be rigorous, fair, \
and actionable.

## Scoring Rubric (0-10 for each dimension)

**Clarity** — Is the language precise and professional? Can a non-expert \
understand the core message? (9-10: flawless prose, zero ambiguity; \
5-6: mostly clear with some jargon issues; 0-3: confusing or poorly written)

**Persuasiveness** — Does the proposal make a compelling case? Are benefits \
quantified? Is there a clear value proposition? (9-10: impossible to say no; \
5-6: reasonable but missing urgency; 0-3: weak or unconvincing)

**Completeness** — Does it address every requirement from the original task? \
Are all necessary sections present? (9-10: comprehensive, nothing missing; \
5-6: covers basics but gaps exist; 0-3: major sections missing)

**Structure** — Is there a logical flow? Are headings, transitions, and \
formatting professional? (9-10: textbook structure; 5-6: adequate but \
could be reordered; 0-3: disorganised)

**Specificity** — Is the content tailored to the client/industry with \
real data, names, and numbers? (9-10: deeply personalised; 5-6: some \
customisation; 0-3: generic boilerplate)

## Rules
- Calculate overall_score as the arithmetic mean of all 5 dimension scores.
- If overall_score >= 9.0, set critique to an empty string.
- If overall_score < 9.0, provide exactly 3 specific, actionable \
improvements the writer should make in the next revision.
- Be strict but constructive — every point of critique should explain \
both *what* is wrong and *how* to fix it."""


# ── Graph node ───────────────────────────────────────────────────────
def evaluator_node(state: AgentState) -> dict:
    """LangGraph node – scores the draft and returns structured feedback."""
    llm = get_llm(model="openai/gpt-oss-120b", temperature=0.1)
    structured_llm = llm.with_structured_output(EvaluationOutput)

    draft = state.get("draft", "")
    task = state.get("task", "")

    prompt = ChatPromptTemplate.from_messages([
        ("system", EVALUATOR_SYSTEM_PROMPT),
        ("human", f"Task: {task}\n\nDraft:\n{draft}"),
    ])

    result: EvaluationOutput = structured_llm.invoke(prompt.format_messages())

    dimension_scores = {
        "Clarity": result.clarity,
        "Persuasiveness": result.persuasiveness,
        "Completeness": result.completeness,
        "Structure": result.structure,
        "Specificity": result.specificity,
    }

    current_revisions = state.get("revision_count", 0)

    return {
        "messages": [AIMessage(content=f"Evaluated draft. Score: {result.overall_score}/10")],
        "score": result.overall_score,
        "critique": result.critique,
        "dimension_scores": dimension_scores,
        "revision_count": current_revisions + 1,
    }
