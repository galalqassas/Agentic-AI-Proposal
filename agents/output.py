"""Output agent – converts the final draft into Markdown files."""

from __future__ import annotations

import os
import time

from langchain_core.messages import AIMessage

from graph.state import AgentState

OUTPUT_DIR = "outputs"

def output_node(state: AgentState) -> dict:
    """LangGraph node – saves the final draft to disk."""
    draft = state.get("draft", "")
    proposal_type = state.get("proposal_type", "General")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base_filename = f"{timestamp}_{proposal_type}_Proposal"
    
    md_path = os.path.join(OUTPUT_DIR, f"{base_filename}.md")
    
    # 1. Save Markdown
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(draft)
        
    # PDF generation skipped due to missing system dependencies (cairo/gtk)
    # on Windows environment without manual installation.
    
    msg = f"Saved proposal to {md_path}"
        
    return {
        "messages": [AIMessage(content=msg)],
        # We don't update state further as this is the end
    }
