# Agentic Classifier Studio

A lightweight, local-first demo of a user-trainable agentic text classifier. Users upload a CSV, label a small seed set, train a classifier, and run it on the remaining rows. The run step uses simple agent orchestration: a trained model (if available), a keyword agent, and an LLM agent (optional) are fused into one final prediction with a self-check adjustment.

## What it does

- Upload a CSV with `text` and `classification`
- Label rows in a simple dashboard
- Train a reusable classifier
- Run predictions on unlabeled rows
- Review low-confidence items (uncertainty queue)

## Architecture (minimal)

- FastAPI backend + Jinja templates
- SQLite for persistence
- TF-IDF + Logistic Regression for training
- Optional LLM agent via OpenAI (LangChain)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` and login with:
- user: `admin`
- password: `1234`

## LLM agent (optional)

Create `.env` with your key:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

If the key is missing, the app falls back to non-LLM agents.

## CSV format

Exactly two columns:

```
text,classification
"sample text",label_a
"another text",
```

Unlabeled rows should have an empty `classification`.

## Typical flow

1) Upload a CSV  
2) Label a handful of rows  
3) Create agent and train  
4) Run predictions  
5) Review uncertain items and correct labels  

## Notes

- This is a demo app, not production hardened.
- `.env` is ignored by git by default.
