"""Field-level F1 + hallucination check against gold labels.

Per-field-type scoring:
- diagnoses/medications/procedures/lab_values/timeline: exact-match F1 over normalized string tuples
- hallucination: fraction of predicted items whose key content appears as a substring in the source note

Scoring is intentionally strict on the identity tuple (name+dose+route for meds; description+icd10 for dx)
but tolerant of surrounding whitespace/case. This mirrors how a chart-abstraction eval would work in production:
the extracted structured record either matches the gold record or it doesn't.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from abstraction.schemas import ChartAbstraction


@dataclass
class FieldScore:
    field_name: str
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class NoteScorecard:
    note_id: str
    field_scores: dict[str, FieldScore] = field(default_factory=dict)
    hallucination_rate: float = 0.0
    n_hallucinated: int = 0
    n_predicted_total: int = 0
    latency_s: float = 0.0
    backend: str = ""

    @property
    def macro_f1(self) -> float:
        if not self.field_scores:
            return 0.0
        return sum(s.f1 for s in self.field_scores.values()) / len(self.field_scores)


def _norm(s: str | None) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _dx_key(d: dict) -> tuple:
    return (_norm(d.get("description")), _norm(d.get("icd10")))


def _med_key(m: dict) -> tuple:
    return (_norm(m.get("name")), _norm(m.get("dose")), _norm(m.get("route")), _norm(m.get("frequency")))


def _proc_key(p: dict) -> tuple:
    return (_norm(p.get("name")), _norm(p.get("date")))


def _lab_key(l: dict) -> tuple:
    return (_norm(l.get("test")), _norm(l.get("value")), _norm(l.get("unit")))


def _tl_key(t: dict) -> tuple:
    return (_norm(t.get("date")), _norm(t.get("event"))[:40])  # 40-char event prefix


_KEY_FUNCS = {
    "diagnoses": _dx_key,
    "medications": _med_key,
    "procedures": _proc_key,
    "lab_values": _lab_key,
    "timeline": _tl_key,
}


def _score_field(name: str, predicted: list[dict], gold: list[dict]) -> FieldScore:
    key_fn = _KEY_FUNCS[name]
    p_keys = {key_fn(x) for x in predicted}
    g_keys = {key_fn(x) for x in gold}
    score = FieldScore(field_name=name)
    score.tp = len(p_keys & g_keys)
    score.fp = len(p_keys - g_keys)
    score.fn = len(g_keys - p_keys)
    return score


def _hallucination_rate(prediction: ChartAbstraction, source_text: str) -> tuple[float, int, int]:
    """Fraction of predicted items whose primary token doesn't appear in the source note.

    Rough proxy for hallucination: checks if the main identifier (dx description, med name, procedure name,
    lab test, timeline event) has substring presence in the source.
    """
    source_lower = source_text.lower()
    predicted_items: list[str] = []
    predicted_items.extend(d.description for d in prediction.diagnoses)
    predicted_items.extend(m.name for m in prediction.medications)
    predicted_items.extend(p.name for p in prediction.procedures)
    predicted_items.extend(lv.test for lv in prediction.lab_values)
    predicted_items.extend(t.event[:20] for t in prediction.timeline)

    if not predicted_items:
        return 0.0, 0, 0

    # normalize: strip punctuation, lowercase, take first 3 words
    hallucinated = 0
    for item in predicted_items:
        words = re.sub(r"[^\w\s]", " ", (item or "").lower()).split()[:3]
        if not words:
            continue
        # count as "grounded" if at least 2 of first 3 words are in source
        matches = sum(1 for w in words if w in source_lower)
        if matches < min(2, len(words)):
            hallucinated += 1
    return hallucinated / len(predicted_items), hallucinated, len(predicted_items)


def score_note(
    note_id: str,
    prediction: ChartAbstraction,
    gold_labels: dict[str, Any],
    source_text: str,
    latency_s: float = 0.0,
    backend: str = "",
) -> NoteScorecard:
    card = NoteScorecard(note_id=note_id, latency_s=latency_s, backend=backend)
    pred_dict = prediction.model_dump()
    for fname in _KEY_FUNCS:
        card.field_scores[fname] = _score_field(fname, pred_dict.get(fname, []), gold_labels.get(fname, []))
    h_rate, n_h, n_total = _hallucination_rate(prediction, source_text)
    card.hallucination_rate = h_rate
    card.n_hallucinated = n_h
    card.n_predicted_total = n_total
    return card


def scorecard_summary(cards: list[NoteScorecard]) -> dict[str, Any]:
    if not cards:
        return {}
    by_field: dict[str, list[float]] = {}
    for c in cards:
        for fname, fs in c.field_scores.items():
            by_field.setdefault(fname, []).append(fs.f1)
    return {
        "n_notes": len(cards),
        "macro_f1": round(sum(c.macro_f1 for c in cards) / len(cards), 3),
        "per_field_f1": {k: round(sum(v) / len(v), 3) for k, v in by_field.items()},
        "hallucination_rate_avg": round(sum(c.hallucination_rate for c in cards) / len(cards), 3),
        "latency_s_avg": round(sum(c.latency_s for c in cards) / len(cards), 2),
        "backends": sorted({c.backend for c in cards if c.backend}),
    }
