"""Thin client for a local OpenAI-compatible LLM endpoint (Ollama / llama.cpp / MLX).

Default targets Ollama at http://localhost:11434/v1. On the M5 Pro (48 GB) a 27-32B
model at Q4 runs comfortably: `ollama pull qwen2.5:32b` then it's ready. The client is a
plain callable so the narrator can be unit-tested with a mock in its place.
"""

from __future__ import annotations

import httpx

from ..config import LLM_BASE_URL, LLM_MODEL


class LocalLLM:
    def __init__(self, base_url: str | None = None, model: str | None = None,
                 timeout: float = 600.0, max_tokens: int | None = None,
                 reasoning_effort: str | None = None):
        """``max_tokens`` caps the COMPLETION length server-side. Interactive callers
        (chat, digest) must set it: an uncapped local model happily writes a
        2,800-token essay to a one-line question, and at ~35 tok/s that alone eats
        the whole request timeout. 'Answer concisely' in a prompt is a wish; this
        is the bound. Long-form narration passes None and stays uncapped.

        ``reasoning_effort`` ("none"/"low"/…) rides along for thinking models —
        measured on gemma4:26b, hidden reasoning burned 300+ tokens on a one-line
        question and, under a cap, could eat the WHOLE budget and return empty
        content. Ollama's OpenAI endpoint honors it; servers that don't ignore it."""
        self.base_url = (base_url or LLM_BASE_URL).rstrip("/")
        self.model = model or LLM_MODEL
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self._client = httpx.Client(timeout=timeout)
        self.last_usage: dict = {}  # token usage of the most recent completion

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if self.max_tokens:
            payload["max_tokens"] = int(self.max_tokens)
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        resp = self._client.post(f"{self.base_url}/chat/completions", json=payload)
        resp.raise_for_status()
        body = resp.json()
        self.last_usage = body.get("usage") or {}
        return body["choices"][0]["message"]["content"]

    def __call__(self, system: str, user: str, **kwargs) -> str:
        return self.complete(system, user, **kwargs)


def make_llm(tier: str = "full") -> LocalLLM:
    """The one place LLM clients are constructed, keyed by usage tier.

    - ``"full"``  — narration: LLM_MODEL, uncapped, long timeout (the read is the point)
    - ``"light"`` — routine minor-change narration + news extraction: LLM_LIGHT_MODEL
    - ``"chat"``  — interactive surfaces (ask / book / explain-move / digest brief):
      the chat model with a hard completion cap, short timeout, and reasoning off
      (config LLM_CHAT_*; see the bench verdicts recorded there)

    Local-only by decision (2026-07-11), not limitation: no cloud providers, no API
    keys — portfolio data never leaves the machine for inference. If a provider is
    ever added, it is one new class honoring the same ``complete(system, user) -> str``
    contract, selected HERE, so the ~dozen call sites never change again.
    """
    from ..config import (
        LLM_CHAT_MAX_TOKENS,
        LLM_CHAT_MODEL,
        LLM_CHAT_REASONING,
        LLM_CHAT_TIMEOUT,
        LLM_LIGHT_MODEL,
    )

    if tier == "chat":
        return LocalLLM(model=LLM_CHAT_MODEL, timeout=LLM_CHAT_TIMEOUT,
                        max_tokens=LLM_CHAT_MAX_TOKENS,
                        reasoning_effort=LLM_CHAT_REASONING)
    if tier == "light":
        return LocalLLM(model=LLM_LIGHT_MODEL)
    if tier == "full":
        return LocalLLM()
    raise ValueError(f"unknown LLM tier: {tier!r} (full | light | chat)")
