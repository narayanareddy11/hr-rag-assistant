# HR RAG Assistant

A beginner-friendly RAG application using LangGraph, ChromaDB, Gemini, and Streamlit.

## Architecture

```text
User
  -> Streamlit
  -> LangGraph classifier/router
  -> HR / Billing ChromaDB retrieval
  -> Gemini 2.5 Flash
  -> Answer
```

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
streamlit run streamlit_app.py
```

## Local setup with uv

```bash
uv sync
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
uv run streamlit run streamlit_app.py
```

## Replit setup

1. Import this GitHub repository into Replit.
2. Add a Replit Secret named `GEMINI_API_KEY` containing your Gemini API key.
3. Run:

```bash
uv lock
uv sync
uv run streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port $PORT
```

The repository contains a `.replit` file with the Streamlit run command, so Replit can use the Run button after dependencies are synchronized.

## Demo questions

- How many days can I work from home?
- How many annual leave days do employees get?
- Can I get a refund for my annual subscription?
- What is Kubernetes?

## Document extraction POC

Sample documents live in `documents/`.

Run the lightweight extraction and keyword-search POC:

```bash
python document_poc.py
```

The POC reads text-like files locally and can use Gemini Vision for `.png`, `.jpg`, `.jpeg`, and `.webp` image documents when `GEMINI_API_KEY` is set. This is a simple image/document extraction demo, not a full ColPali retrieval pipeline.

## Security

Never commit API keys, `.env` files, or private credentials to GitHub. Use Replit Secrets or environment variables.

If the Gemini API key is expired, invalid, missing, or out of quota, the app shows a clear `Gemini API key/token expired or invalid` message.
