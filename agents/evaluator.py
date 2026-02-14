"""Evaluator agent – scores the proposal draft and provides critique.

The evaluator:
1. Reads the draft and the original task.
2. Scores it on 5 dimensions (0-10).
3. Calculates an average score.
4. Generates a critique if the score is below threshold.
"""

from __future__ import annotations

import re
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from graph.state import AgentState
from utils.llm import get_llm

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
(Only if Overall Score < 8.0)
Provide exactly 3 specific improvements needed. 
If the Overall Score is 8.0 or higher, output: None
"""

def evaluator_node(state: AgentState) -> dict:
    """LangGraph node – scores the draft."""
    llm = get_llm(temperature=0.1)
    
    draft = state.get("draft", "")
    task = state.get("task", "")
    
    # Construct the evaluation prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", EVALUATOR_SYSTEM_PROMPT),
        ("human", f"Task: {task}\n\nDraft:\n{draft}")
    ])
    
    response = llm.invoke(prompt.format_messages())
    content = response.content if hasattr(response, "content") else str(response)
    
    score = _extract_overall_score(content)
    critique = _extract_critique(content)
    
    # Increment revision count
    current_revisions = state.get("revision_count", 0)
    
    return {
        "messages": [AIMessage(content=f"Evaluated draft. Score: {score}/10")],
        "score": score,
        "critique": critique,
        "revision_count": current_revisions + 1
    }

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
