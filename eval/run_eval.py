"""Run the full eval harness: extract all notes, score against gold, print scorecard.

Usage:
    python -m eval.run_eval                    # all notes
    python -m eval.run_eval --notes note_001   # single note
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from abstraction.extractor import extract
from eval.scorer import score_note, scorecard_summary

REPO = Path(__file__).parent.parent
NOTES_DIR = REPO / "data" / "notes"
GOLD_DIR = REPO / "data" / "gold"


def run(note_ids: list[str] | None = None) -> None:
    all_notes = sorted(NOTES_DIR.glob("*.txt"))
    if note_ids:
        all_notes = [n for n in all_notes if n.stem in note_ids]
    if not all_notes:
        print("no notes found"); return

    cards = []
    for np in all_notes:
        note_text = np.read_text()
        gold_path = GOLD_DIR / f"{np.stem}.json"
        if not gold_path.exists():
            print(f"skip {np.stem} (no gold)")
            continue
        gold = json.loads(gold_path.read_text())

        print(f"\n=== {np.stem} ===")
        prediction, telemetry = extract(note_text)
        print(f"  backend={telemetry['backend']}  latency={telemetry['latency_s']}s  retries={telemetry['retries']}")
        if telemetry.get("parse_error"):
            print(f"  PARSE ERROR: {telemetry['parse_error']}")

        card = score_note(np.stem, prediction, gold, note_text, telemetry["latency_s"], telemetry["backend"])
        cards.append(card)
        print(f"  macro F1: {card.macro_f1:.3f}  hallucination rate: {card.hallucination_rate:.2%}")
        for fname, fs in card.field_scores.items():
            print(f"    {fname:<12}  F1={fs.f1:.2f}  P={fs.precision:.2f}  R={fs.recall:.2f}  (tp={fs.tp} fp={fs.fp} fn={fs.fn})")

    if cards:
        print("\n=== SUMMARY ===")
        print(json.dumps(scorecard_summary(cards), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--notes", nargs="*", help="Note IDs to eval (default: all)")
    args = p.parse_args()
    run(args.notes)
