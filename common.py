"""Shared helpers: load the class .env and build an OpenRouter chat model for LangChain."""
from __future__ import annotations

import os
from pathlib import Path

from langchain_openai import ChatOpenAI

AGENT_MODEL = os.environ.get("AGENT_MODEL", "anthropic/claude-haiku-5.5")  # benchmark default


def load_env(path: str | os.PathLike | None = None) -> None:
    """Load KEY=VALUE lines into os.environ (existing variables win)."""
    path = Path(path or Path(__file__).resolve().parents[2] / ".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip())


def tracing_enabled() -> bool:
    """Langfuse tracing is on only when both keys are configured."""
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def langfuse_callbacks(trace_context: dict | None = None) -> list:
    """LangChain callbacks that send the run to Langfuse (empty list when tracing is off)."""
    if not tracing_enabled():
        return []
    from langfuse.langchain import CallbackHandler
    return [CallbackHandler(trace_context=trace_context)] if trace_context else [CallbackHandler()]


def llm(model: str = AGENT_MODEL, max_tokens: int = 1024) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0,
        max_tokens=max_tokens,
        timeout=120,
        max_retries=2,
        default_headers={"X-Title": "agents-class-a2a-prototype"},
    )
