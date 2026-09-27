"""Streamlit UI: paste a clinical note, get structured extraction + eval scorecard.

Layout:
- Sidebar: backend indicator, 3 preloaded example notes, run-full-eval button
- Main: text area for note, extraction results tabbed by field, telemetry expander
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from abstraction import llm  # noqa: E402
from abstraction.extractor import extract  # noqa: E402
from eval.scorer import score_note, scorecard_summary  # noqa: E402

NOTES_DIR = ROOT / "data" / "notes"
GOLD_DIR = ROOT / "data" / "gold"

st.set_page_config(page_title="Chart Abstraction LLM Eval", page_icon=":material/health_and_safety:", layout="wide")

# ---- sidebar ----
with st.sidebar:
    st.title(":material/medical_information: Chart Abstraction")

    backend = llm.backend_name()
    status_icon = ":material/check_circle:" if backend != "mock" else ":material/warning:"
    st.caption(f"{status_icon} LLM backend: **{backend}**")
    if backend == "mock":
        st.warning("Set `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY` / `NVIDIA_API_KEY`) in `.env` or Streamlit secrets for real extraction.")

    st.divider()
    st.subheader("Load an example note")
    notes = sorted(NOTES_DIR.glob("*.txt"))
    for np in notes:
        if st.button(np.stem, key=f"n_{np.stem}", icon=":material/description:", use_container_width=True):
            st.session_state.note_text = np.read_text()
            st.session_state.selected_note = np.stem

    st.divider()
    if st.button("Run full eval (all notes)", type="primary", icon=":material/play_arrow:", use_container_width=True):
        st.session_state.run_full_eval = True


# ---- main pane ----
st.title("Chart Abstraction from Clinical Notes")
st.caption("LLM-powered structured extraction · PrismBench-style eval harness · Synthetic MIMIC-shaped data")

with st.expander("What this is", expanded=False):
    st.markdown("""
Reads unstructured clinical notes (discharge summaries) and extracts **5 structured fields**:

1. **Diagnoses** (with ICD-10 codes)
2. **Medications** (name, dose, route, frequency)
3. **Procedures** (name, date)
4. **Lab values** (test, value, unit, date)
5. **Timeline** (dated clinical events)

**Eval harness scores each extraction against gold labels:**
- Per-field F1 (precision + recall on exact-match normalized tuples)
- Hallucination rate (fraction of predicted items whose primary token isn't in source text)
- Latency + backend telemetry

Synthetic notes are used (no PHI, no MIMIC-DUA required). Same pipeline runs on real MIMIC-III / n2c2 with only path changes.
""")

# ---- note input ----
default_note = st.session_state.get("note_text", "")
note_text = st.text_area(
    "Clinical note",
    value=default_note,
    height=300,
    placeholder="Paste a discharge summary or clinical note here, or click an example in the sidebar.",
)

col_extract, col_eval = st.columns([1, 1])
selected_note = st.session_state.get("selected_note")
run_extract = col_extract.button(
    "Extract structured fields",
    type="primary",
    icon=":material/biotech:",
    disabled=not note_text.strip(),
    use_container_width=True,
)
run_eval_this = col_eval.button(
    f"Extract + score vs. gold" + (f" ({selected_note})" if selected_note else ""),
    icon=":material/scoreboard:",
    disabled=not (selected_note and (GOLD_DIR / f"{selected_note}.json").exists()),
    use_container_width=True,
)

# ---- run single extraction ----
if run_extract or run_eval_this:
    with st.spinner("Extracting…"):
        prediction, telemetry = extract(note_text)

    st.success(f"Backend: {telemetry['backend']} · Latency: {telemetry['latency_s']}s · Retries: {telemetry['retries']}")

    tabs = st.tabs(["Diagnoses", "Medications", "Procedures", "Lab Values", "Timeline"])
    with tabs[0]:
        st.dataframe(pd.DataFrame([d.model_dump() for d in prediction.diagnoses]) if prediction.diagnoses else pd.DataFrame(), use_container_width=True)
    with tabs[1]:
        st.dataframe(pd.DataFrame([m.model_dump() for m in prediction.medications]) if prediction.medications else pd.DataFrame(), use_container_width=True)
    with tabs[2]:
        st.dataframe(pd.DataFrame([p.model_dump() for p in prediction.procedures]) if prediction.procedures else pd.DataFrame(), use_container_width=True)
    with tabs[3]:
        st.dataframe(pd.DataFrame([l.model_dump() for l in prediction.lab_values]) if prediction.lab_values else pd.DataFrame(), use_container_width=True)
    with tabs[4]:
        st.dataframe(pd.DataFrame([t.model_dump() for t in prediction.timeline]) if prediction.timeline else pd.DataFrame(), use_container_width=True)

    with st.expander(":material/search: Telemetry (raw LLM response)"):
        st.code(telemetry.get("raw_response", "")[:4000], language="json")

    if run_eval_this and selected_note:
        gold = json.loads((GOLD_DIR / f"{selected_note}.json").read_text())
        card = score_note(selected_note, prediction, gold, note_text, telemetry["latency_s"], telemetry["backend"])
        st.divider()
        st.subheader(f"Scorecard vs. gold ({selected_note})")
        c1, c2, c3 = st.columns(3)
        c1.metric("Macro F1", f"{card.macro_f1:.2%}")
        c2.metric("Hallucination rate", f"{card.hallucination_rate:.2%}")
        c3.metric("Latency", f"{card.latency_s:.1f}s")
        st.dataframe(pd.DataFrame([{
            "field": f.field_name, "F1": round(f.f1, 3), "precision": round(f.precision, 3),
            "recall": round(f.recall, 3), "tp": f.tp, "fp": f.fp, "fn": f.fn,
        } for f in card.field_scores.values()]), use_container_width=True)

# ---- full-eval run ----
if st.session_state.get("run_full_eval"):
    st.session_state.run_full_eval = False
    st.divider()
    st.subheader("Full eval — all notes")
    all_notes = sorted(NOTES_DIR.glob("*.txt"))
    cards = []
    progress = st.progress(0.0)
    for i, np in enumerate(all_notes):
        gold_path = GOLD_DIR / f"{np.stem}.json"
        if not gold_path.exists():
            continue
        with st.spinner(f"Extracting {np.stem}…"):
            note_text_i = np.read_text()
            prediction, telemetry = extract(note_text_i)
            card = score_note(np.stem, prediction, json.loads(gold_path.read_text()), note_text_i, telemetry["latency_s"], telemetry["backend"])
            cards.append(card)
        progress.progress((i + 1) / len(all_notes))
    if cards:
        st.success("Done.")
        summary = scorecard_summary(cards)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Notes scored", summary["n_notes"])
        c2.metric("Macro F1 avg", f"{summary['macro_f1']:.2%}")
        c3.metric("Hallucination avg", f"{summary['hallucination_rate_avg']:.2%}")
        c4.metric("Latency avg", f"{summary['latency_s_avg']:.1f}s")
        st.subheader("Per-field F1")
        st.bar_chart(pd.DataFrame(summary["per_field_f1"].items(), columns=["field", "F1"]).set_index("field"))
        st.subheader("Per-note scorecard")
        st.dataframe(pd.DataFrame([{
            "note_id": c.note_id, "macro_F1": round(c.macro_f1, 3),
            "hallucination_rate": round(c.hallucination_rate, 3), "latency_s": c.latency_s, "backend": c.backend,
        } for c in cards]), use_container_width=True)
