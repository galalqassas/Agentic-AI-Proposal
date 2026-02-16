"""LLM initialization utility.

Loads GROQ_API_KEY and GROQ_MODEL from the .env file and
exposes a single `get_llm()` helper that returns a ChatGroq instance.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-120b"


def get_llm(model: str | None = None, temperature: float = 0.3) -> ChatGroq:
    """Return a ChatGroq instance configured from environment variables.

    Parameters
    ----------
    model : str, optional
        Groq model ID to use (e.g. "openai/gpt-oss-120b", "groq/compound").
        Falls back to GROQ_MODEL env var, then DEFAULT_MODEL.
    temperature : float
        Sampling temperature (default 0.3).
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    resolved_model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)

    return ChatGroq(
        model=resolved_model,
        api_key=api_key,
        temperature=temperature,
    )
