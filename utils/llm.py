"""LLM initialization utility.

Loads OLLAMA_MODEL and OLLAMA_BASE_URL from the .env file and
exposes a single `get_llm()` helper that returns a ChatOllama instance.
"""

import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

DEFAULT_MODEL = "gpt-oss:20b"
DEFAULT_BASE_URL = "http://localhost:11434"


def get_llm(temperature: float = 0.3) -> ChatOllama:
    """Return a ChatOllama instance configured from environment variables."""
    model = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)

    # Strip common suffixes if present
    base_url = base_url.rstrip("/")
    for suffix in ["/v1/chat/completions", "/v1"]:
        if base_url.endswith(suffix):
            base_url = base_url[: -len(suffix)]
            break

    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
    )
