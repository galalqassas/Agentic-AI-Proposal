"""Tests for the Utils module."""

import os
from unittest.mock import patch, MagicMock
from utils.llm import get_llm
from utils.templates import PROPOSAL_TEMPLATES

class TestLLM:
    """Tests for llm.py."""

    @patch("utils.llm.ChatGroq")
    @patch.dict(os.environ, {"GROQ_API_KEY": "test_key", "GROQ_MODEL": "test_model"})
    def test_get_llm_configures_correctly(self, mock_chat_groq):
        """Should initialize ChatGroq with env vars."""
        get_llm(temperature=0.5)
        
        mock_chat_groq.assert_called_once()
        _, kwargs = mock_chat_groq.call_args
        assert kwargs["api_key"] == "test_key"
        assert kwargs["model"] == "test_model"
        assert kwargs["temperature"] == 0.5

    @patch("utils.llm.ChatGroq")
    def test_get_llm_defaults(self, mock_chat_groq):
        """Should use defaults when env vars missing (and args not provided)."""
        # We assume DEFAULT_MODEL is defined in llm.py
        from utils.llm import DEFAULT_MODEL
        
        # Clear env vars for this test
        with patch.dict(os.environ, {}, clear=True):
            get_llm()
            
            _, kwargs = mock_chat_groq.call_args
            assert kwargs["model"] == DEFAULT_MODEL

class TestTemplates:
    """Tests for templates.py."""

    def test_templates_exist(self):
        """Standard proposal types should have templates."""
        expected_keys = [
            "Business", "Grant", "Technical", "Sales", 
            "Project", "Research", "Partnership", "General"
        ]
        for key in expected_keys:
            assert key in PROPOSAL_TEMPLATES
            assert "{plan}" in PROPOSAL_TEMPLATES[key]
            assert "{research_data}" in PROPOSAL_TEMPLATES[key]
