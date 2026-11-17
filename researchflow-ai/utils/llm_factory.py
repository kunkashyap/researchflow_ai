"""
utils/llm_factory.py
Unified LLM factory supporting OpenAI, Google Gemini, and OpenAI-compatible providers.
"""

from __future__ import annotations

import os
from typing import Optional
from utils.config import config
from utils.logging_config import setup_logging

logger = setup_logging("researchflow.llm_factory")


def create_llm(
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
    model_name: Optional[str] = None,
):
    """
    Creates and returns a LangChain Chat Model based on available API keys.
    Automatically prioritises Google Gemini if GEMINI_API_KEY is set.
    """
    gemini_key = (
        getattr(config, "GEMINI_API_KEY", "")
        or os.getenv("GEMINI_API_KEY", "")
        or os.getenv("GOOGLE_API_KEY", "")
    )
    openai_key = (
        getattr(config, "OPENAI_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
    )

    has_gemini = bool(gemini_key and gemini_key != "your_gemini_api_key_here")
    has_openai = bool(openai_key and openai_key != "your_openai_api_key_here")

    llm_provider = getattr(config, "LLM_PROVIDER", "auto").lower()

    # Prioritise Gemini if key exists and OpenAI is not set, or if provider explicitly set to gemini
    if has_gemini and (not has_openai or llm_provider == "gemini"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            gemini_model = getattr(config, "GEMINI_MODEL", "gemini-flash-latest")
            model = model_name or gemini_model or "gemini-flash-latest"
            if "gpt-" in model:
                model = gemini_model or "gemini-flash-latest"

            kwargs = {
                "model": model,
                "temperature": temperature,
                "google_api_key": gemini_key,
            }
            if max_tokens:
                kwargs["max_output_tokens"] = max_tokens

            logger.info("Initializing ChatGoogleGenerativeAI with model: %s", model)
            return ChatGoogleGenerativeAI(**kwargs)
        except Exception as exc:
            logger.warning(
                "Could not initialize ChatGoogleGenerativeAI (%s); falling back to ChatOpenAI", exc
            )

    from langchain_openai import ChatOpenAI

    openai_model = getattr(config, "OPENAI_MODEL", "gpt-4o-mini")
    openai_base_url = getattr(config, "OPENAI_BASE_URL", "")

    kwargs = {
        "model": model_name or openai_model,
        "temperature": temperature,
        "openai_api_key": openai_key,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if openai_base_url:
        kwargs["base_url"] = openai_base_url

    logger.info("Initializing ChatOpenAI with model: %s", kwargs["model"])
    return ChatOpenAI(**kwargs)
