"""LLM backend abstraction. Anthropic / OpenAI / NVIDIA NIM / mock."""
from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

BACKEND = (os.getenv("LLM_BACKEND") or "auto").lower()
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
NVIDIA_KEY = os.getenv("NVIDIA_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")


def _pick() -> str:
    if BACKEND != "auto":
        return BACKEND
    if ANTHROPIC_KEY:
        return "anthropic"
    if OPENAI_KEY:
        return "openai"
    if NVIDIA_KEY:
        return "nvidia"
    return "mock"


def backend_name() -> str:
    return _pick()


def chat(system: str, user: str, temperature: float = 0.0) -> str:
    """Return raw text response from the chosen backend."""
    b = _pick()
    if b == "anthropic":
        return _anthropic(system, user, temperature)
    if b == "openai":
        return _openai(system, user, temperature)
    if b == "nvidia":
        return _nvidia(system, user, temperature)
    return _mock(system, user)


def _anthropic(system: str, user: str, temperature: float) -> str:
    from anthropic import Anthropic, APIError

    client = Anthropic(api_key=ANTHROPIC_KEY)
    try:
        r = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=4096,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except APIError as e:
        raise RuntimeError(
            f"Anthropic API error for model '{ANTHROPIC_MODEL}': {e}. "
            f"If the model slug is wrong, try one of the currently-live IDs: "
            f"'claude-sonnet-4-5-20250929' (default, recommended), "
            f"'claude-opus-4-1-20250805' (highest quality, higher cost), "
            f"'claude-haiku-4-5-20251001' (fastest, cheapest). "
            f"Set ANTHROPIC_MODEL env var to override."
        ) from e
    return "".join(b.text for b in r.content if b.type == "text")


def _openai(system: str, user: str, temperature: float) -> str:
    from openai import OpenAI
    r = OpenAI(api_key=OPENAI_KEY).chat.completions.create(
        model=OPENAI_MODEL,
        temperature=temperature,
        max_tokens=4096,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return r.choices[0].message.content or ""


def _nvidia(system: str, user: str, temperature: float) -> str:
    import requests
    r = requests.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {NVIDIA_KEY}", "Content-Type": "application/json"},
        json={
            "model": NVIDIA_MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": 4096,
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def _mock(system: str, user: str) -> str:
    """Deterministic mock: returns a minimal valid JSON for any input."""
    return json.dumps({
        "diagnoses": [{"description": "[mock] Set ANTHROPIC_API_KEY for real extraction", "icd10": None}],
        "medications": [],
        "procedures": [],
        "lab_values": [],
        "timeline": [],
    })
