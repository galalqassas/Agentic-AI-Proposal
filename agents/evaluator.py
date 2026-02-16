"""Evaluator agent – scores the proposal draft and provides critique.

The evaluator:
1. Reads the draft and the original task.
2. Scores it on 5 dimensions (0-10).
3. Calculates an average score.
4. Generates a critique if the score is below threshold.
"""

from __future__ import annotations

import re
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from graph.state import AgentState
from utils.llm import get_llm

# ── Constants ────────────────────────────────────────────────────────
SCORE_DIMENSIONS = ("Clarity", "Persuasiveness", "Completeness", "Structure", "Specificity")

EVALUATOR_SYSTEM_PROMPT = """\
You are a strict proposal evaluator. Review the draft below against the original task.

Score the draft from 0 to 10 on these 5 dimensions:
1. Clarity (Clear language, easy to read)
2. Persuasiveness (Compelling arguments, benefits-focused)
3. Completeness (Addresses all task requirements)
4. Structure (Logical flow, proper formatting)
5. Specificity (Customised to the client/industry, not generic)

Output format (exactly as shown):
### Scores
Clarity: <score>
Persuasiveness: <score>
Completeness: <score>
Structure: <score>
Specificity: <score>

### Overall Score
<average_score>

### Critique
(Only if Overall Score < 9.5)
Provide exactly 3 specific improvements needed. 
If the Overall Score is 9.5 or higher, output: None
"""


# ── Graph node ───────────────────────────────────────────────────────
def evaluator_node(state: AgentState) -> dict:
    """LangGraph node – scores the draft and returns structured feedback."""
    llm = get_llm(model="openai/gpt-oss-120b", temperature=0.1)

    draft = state.get("draft", "")
    task = state.get("task", "")

    prompt = ChatPromptTemplate.from_messages([
        ("system", EVALUATOR_SYSTEM_PROMPT),
        ("human", f"Task: {task}\n\nDraft:\n{draft}"),
    ])

    response = llm.invoke(prompt.format_messages())
    content = response.content if hasattr(response, "content") else str(response)

    score = _extract_overall_score(content)
    critique = _extract_critique(content)
    dimension_scores = _extract_dimension_scores(content)

    current_revisions = state.get("revision_count", 0)

    return {
        "messages": [AIMessage(content=f"Evaluated draft. Score: {score}/10")],
        "score": score,
        "critique": critique,
        "dimension_scores": dimension_scores,
        "revision_count": current_revisions + 1,
    }


# ── Helpers ──────────────────────────────────────────────────────────
def _extract_overall_score(text: str) -> float:
    """Extract the overall score from the evaluation output."""
    match = re.search(r"### Overall Score[:\s]*(\d+(\.\d+)?)", text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return 0.0


def _extract_critique(text: str) -> str:
    """Extract the critique section."""
    match = re.search(r"### Critique\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _extract_dimension_scores(text: str) -> dict:
    """Extract individual dimension scores into a dict.

    Returns a mapping like ``{"Clarity": 8.0, "Persuasiveness": 7.5, ...}``.
    Missing dimensions default to ``0.0``.
    """
    scores: dict[str, float] = {}
    for dim in SCORE_DIMENSIONS:
        pattern = rf"{dim}[:\s]*(\d+(?:\.\d+)?)"
        match = re.search(pattern, text, re.IGNORECASE)
        scores[dim] = float(match.group(1)) if match else 0.0
    return scores
