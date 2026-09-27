"""Structured extraction of the 5 chart-abstraction targets from a clinical note.

Single LLM call with:
- Explicit JSON schema in the system prompt (self-documenting output contract)
- 1-shot exemplar based on note_001 gold labels (grounds the model on our format)
- Chain-of-thought scaffolding in the prompt ('Think step by step, then output only JSON')
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from . import llm
from .schemas import SCHEMA_JSON_HINT, ChartAbstraction

REPO_ROOT = Path(__file__).parent.parent
EXEMPLAR_NOTE = REPO_ROOT / "data" / "notes" / "note_001.txt"
EXEMPLAR_GOLD = REPO_ROOT / "data" / "gold" / "note_001.json"


def _system_prompt() -> str:
    exemplar_note = EXEMPLAR_NOTE.read_text() if EXEMPLAR_NOTE.exists() else "(exemplar unavailable)"
    exemplar_gold = EXEMPLAR_GOLD.read_text() if EXEMPLAR_GOLD.exists() else "{}"

    return f"""You are a clinical chart-abstraction assistant. Your task is to read a discharge summary or clinical note and extract structured fields.

OUTPUT CONTRACT (JSON only, no prose, no markdown fences):
The response MUST be valid JSON matching this Pydantic schema:

{json.dumps(SCHEMA_JSON_HINT, indent=2)}

RULES:
1. Only extract facts that are STATED in the note. Do not infer, guess, or hallucinate.
2. If a field is not stated, leave it as null (single field) or empty list (list field).
3. Prefer generic medication names over brand names.
4. Dates must be ISO format YYYY-MM-DD. If only relative dates given (e.g., "day 2"), skip.
5. Numeric lab values as strings (preserve original precision).
6. Timeline should list distinct clinical events with dates; avoid duplicating the medication list.

EXEMPLAR (for format only — do NOT reuse content unless it appears in the target note):

INPUT:
{exemplar_note}

OUTPUT:
{exemplar_gold}
"""


def _extract_json_block(text: str) -> str:
    """Some models wrap JSON in ```json fences. Strip them."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        return m.group(1)
    # first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def extract(note_text: str) -> tuple[ChartAbstraction, dict[str, Any]]:
    """Run the extraction. Returns (structured, telemetry).

    telemetry keys: backend, latency_s, retries, raw_response, parse_error.
    """
    system = _system_prompt()
    user = f"CLINICAL NOTE:\n\n{note_text}\n\nExtract the 5 fields and return JSON only."

    t0 = time.time()
    raw = llm.chat(system=system, user=user, temperature=0.0)
    latency = time.time() - t0

    telemetry: dict[str, Any] = {
        "backend": llm.backend_name(),
        "latency_s": round(latency, 2),
        "retries": 0,
        "raw_response": raw,
        "parse_error": None,
    }

    try:
        parsed = json.loads(_extract_json_block(raw))
        return ChartAbstraction(**parsed), telemetry
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        # one retry with correction prompt
        telemetry["retries"] = 1
        correction = f"Your previous response failed to parse: {e}. Return ONLY valid JSON matching the schema, no prose, no fences."
        raw2 = llm.chat(system=system, user=user + "\n\n" + correction, temperature=0.0)
        telemetry["raw_response"] = raw2
        try:
            parsed = json.loads(_extract_json_block(raw2))
            return ChartAbstraction(**parsed), telemetry
        except Exception as e2:
            telemetry["parse_error"] = str(e2)
            return ChartAbstraction(), telemetry
