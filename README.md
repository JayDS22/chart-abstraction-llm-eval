# Chart Abstraction LLM Eval

**LLM-powered structured extraction from unstructured clinical notes, with a PrismBench-shape evaluation harness.** Reads discharge summaries and extracts 5 structured fields (diagnoses, medications, procedures, lab values, timeline), scores each extraction against gold labels, tracks hallucination + latency + cost.

**Live demo:** _(deploy to Streamlit Cloud or HF Spaces — instructions below)_
**Source:** `github.com/JayDS22/chart-abstraction-llm-eval`

## Why this exists

Chart abstraction (reading unstructured medical text and pulling out structured facts) is the core loop of clinical AI platforms. This repo is a reference implementation of:

1. An **LLM extraction pipeline** with explicit JSON schema, few-shot exemplar, retry-on-parse-failure
2. A **rigorous eval harness** — per-field F1, hallucination-grounding check, latency + backend telemetry
3. A **Streamlit UI** for interactive extraction + batch eval
4. **Synthetic clinical notes** (no PHI, no MIMIC DUA required) — same pipeline runs on real MIMIC-III / n2c2 with only path changes

Built as a portable reference for clinical NLP evaluation workflows.

## The 5 extraction targets

Modelled as Pydantic schemas in [`abstraction/schemas.py`](abstraction/schemas.py):

| Field | Contents |
|---|---|
| Diagnoses | Description + ICD-10 code |
| Medications | Name + dose + route + frequency |
| Procedures | Name + date |
| Lab values | Test + value + unit + date |
| Timeline | Dated clinical events |

## Eval methodology

Per note:
- **Field F1** (exact match on normalized identity tuples: e.g. `(name, dose, route, frequency)` for meds)
- **Hallucination rate**: fraction of predicted items whose primary token isn't found in the source text
- **Latency** and **backend** telemetry

Full-corpus summary reports macro-F1, per-field F1, hallucination avg, and latency avg. See [`eval/scorer.py`](eval/scorer.py) for the exact scoring logic.

## Quick start

```bash
git clone https://github.com/JayDS22/chart-abstraction-llm-eval
cd chart-abstraction-llm-eval
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: set ANTHROPIC_API_KEY (or OPENAI_API_KEY / NVIDIA_API_KEY)

# Run the eval harness on all 3 synthetic notes
python -m eval.run_eval

# Or launch the UI
streamlit run app/streamlit_app.py
```

Without an API key, the app + eval both run against a mock backend that returns placeholder JSON. Set at least one key for real extraction.

## Deploy to Streamlit Cloud

1. Push to a public GitHub repo (this one)
2. Go to https://share.streamlit.io/, sign in with GitHub
3. `New app` → pick this repo → main file: `app/streamlit_app.py`
4. Add `ANTHROPIC_API_KEY` (or another) in Advanced Settings → Secrets
5. Deploy → live URL in ~2 min

## Deploy to Hugging Face Spaces (Docker SDK)

```bash
hf auth login
hf repos create chart-abstraction-llm-eval --repo-type space --space-sdk docker
git remote add hf https://huggingface.co/spaces/<user>/chart-abstraction-llm-eval
git push hf main
# Add ANTHROPIC_API_KEY as a Repository Secret in Space settings
```

## Repo layout

```
chart-abstraction-llm-eval/
├── README.md
├── requirements.txt
├── Dockerfile
├── abstraction/
│   ├── schemas.py       # 5 Pydantic models
│   ├── llm.py           # Anthropic / OpenAI / NVIDIA / mock backend
│   └── extractor.py     # Prompt + JSON parse + retry
├── eval/
│   ├── scorer.py        # Field F1 + hallucination + telemetry
│   └── run_eval.py      # CLI harness
├── app/
│   └── streamlit_app.py # Interactive UI + batch eval
└── data/
    ├── notes/           # 3 synthetic discharge summaries
    └── gold/            # Gold-label JSON per note
```

## Extending to real data

To swap synthetic data for MIMIC-III:

1. Complete CITI training + sign the MIMIC-III DUA at https://physionet.org/content/mimiciii/1.4/
2. Drop discharge summary `.txt` files in `data/notes/`
3. Generate gold labels (manual or LLM-bootstrapped + human-verified) in `data/gold/`
4. Run `python -m eval.run_eval`

The pipeline is data-schema-agnostic — as long as the note file basenames match the gold JSON basenames, the harness runs.

## Author

**Jay Guwalani** · [LinkedIn](https://linkedin.com/in/j-guwalani) · [GitHub](https://github.com/JayDS22) · [Portfolio](https://jayds22.github.io/)

Built as a reference for clinical NLP evaluation. Sister to [`replit-agent-bench`](https://github.com/JayDS22/replit-agent-bench) and [`nvidia-semiconductor-mcp-agentic-platform`](https://github.com/JayDS22/nvidia-semiconductor-mcp-agentic-platform).
