"""Tests for the Output agent."""

from __future__ import annotations

import os
import shutil

from agents.output import output_node, OUTPUT_DIR


class TestOutputNode:
    """Tests for the output_node."""

    def setup_method(self):
        """Clean up output dir before test."""
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)

    def teardown_method(self):
        """Clean up output dir after test."""
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)

    def test_creates_md_only(self):
        """Should create .md file."""
        state = {
            "messages": [],
            "task": "Test task",
            "proposal_type": "Business",
            "draft": "# Test Proposal\n\nThis is a test.",
            "revision_count": 0,
            "plan": "",
            "research_data": "",
            "search_queries": [],
            "critique": "",
            "score": 0.0,
            "dimension_scores": {},
            "user_feedback": "",
            "questions_for_user": [],
        }

        output_node(state)

        # Check files exist
        assert os.path.exists(OUTPUT_DIR)
        files = os.listdir(OUTPUT_DIR)
        md_files = [f for f in files if f.endswith(".md")]

        assert len(md_files) == 1
        assert "Business_Proposal" in md_files[0]

        # Check MD content
        with open(os.path.join(OUTPUT_DIR, md_files[0]), "r") as f:
            content = f.read()
            assert "# Test Proposal" in content
