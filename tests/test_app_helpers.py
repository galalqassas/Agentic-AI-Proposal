"""Tests for the App helper functions."""

import sys
import pytest
from unittest.mock import MagicMock, patch
import importlib

@pytest.fixture(scope="module")
def app_module():
    """Safely import app module with mocks, then cleanup."""
    # Context manager for sys.modules doesn't exist directly, but patch.dict works
    with patch.dict(sys.modules, {
        "chainlit": MagicMock(),
        "chainlit.types": MagicMock(),
        "data_layer": MagicMock(),
        "graph.graph": MagicMock(),
    }):
        # We need to ensure we don't pick up a cached app module
        if "app" in sys.modules:
            del sys.modules["app"]
            
        import app
        yield app
        
        # Cleanup: remove app from sys.modules so it doesn't pollute
        if "app" in sys.modules:
            del sys.modules["app"]

def test_score_emoji(app_module):
    """Should return correct emoji based on score."""
    assert app_module._score_emoji(9.5) == "🟢"
    assert app_module._score_emoji(9.0) == "🟢"
    assert app_module._score_emoji(8.9) == "🟡"
    assert app_module._score_emoji(7.0) == "🟡"
    assert app_module._score_emoji(6.9) == "🔴"
    assert app_module._score_emoji(0.0) == "🔴"

def test_build_scorecard(app_module):
    """Should build a markdown scorecard."""
    dimension_scores = {
        "Clarity": 9.0,
        "Persuasiveness": 6.0
    }
    scorecard = app_module._build_scorecard(dimension_scores, 8.0, 1)
    
    assert "🟢" in scorecard  # Clarity
    assert "🔴" in scorecard  # Persuasiveness
    assert "**Clarity**" not in scorecard # It puts value in bold: **9.0**
    assert "**9.0**" in scorecard
    assert "Overall: 8.0/10" in scorecard
    assert "Revision 1/" in scorecard

def test_writer_label(app_module):
    """Should format writer label."""
    assert app_module._writer_label(1) == "Writer (attempt 1)"
    assert app_module._writer_label(2) == "Writer (attempt 2)"

def test_evaluator_label(app_module):
    """Should format evaluator label."""
    assert app_module._evaluator_label(1) == "Evaluator (attempt 1)"
    assert app_module._evaluator_label(3) == "Evaluator (attempt 3)"
