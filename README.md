# SHL Assessment Recommender

A stateless FastAPI conversational agent that helps hiring managers find the right SHL assessments. Each `POST /chat` request sends the full conversation history; the server uses ChromaDB semantic search and Google Gemini 1.5 Flash to clarify, recommend, refine, compare, or refuse off-topic queries.

## Prerequisites

- Python 3.11+
- `catalog.json` in the project root
- A [Google AI Studio](https://aistudio.google.com/) API key for Gemini

## Setup

1. Create a virtual environment (recommended) and install dependencies:

```bash
pip install -r requirements.txt
```

2. Place `catalog.json` in the project root (already included in this repo).

3. Build the ChromaDB vector index (safe to run multiple times):

```bash
python build_index.py
```

4. Configure your API key securely via environment variable (never commit `.env`):

```bash
# Linux / macOS
export GEMINI_API_KEY=your_key_here

# Windows PowerShell
$env:GEMINI_API_KEY="your_key_here"
```

Alternatively, create a `.env` file in the project root (see `.env.example`). The variable name must be **`GEMINI_API_KEY`**. The app loads it with `python-dotenv` at startup.

5. Run the application:

```bash
python -m uvicorn main:app --reload
```

6. Open the chat UI in your browser:

**http://127.0.0.1:8000**

The same server serves the API (`/health`, `/chat`) and the web frontend.

## API

### `GET /health`

Returns `{"status": "ok"}` with HTTP 200.

### `POST /chat`

**Request:**

```json
{
  "messages": [
    {"role": "user", "content": "I need an assessment for a senior Java developer"}
  ]
}
```

**Response:**

```json
{
  "reply": "...",
  "recommendations": [
    {
      "name": "...",
      "url": "https://www.shl.com/...",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

- `recommendations` is `[]` when clarifying, refusing, or comparing; otherwise 1–10 items.
- All URLs come from `catalog.json` only.

## Test locally

Health check:

```bash
curl http://localhost:8000/health
```

Vague query (should clarify, empty recommendations):

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"messages\": [{\"role\": \"user\", \"content\": \"I need an assessment\"}]}"
```

Specific hiring need:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"messages\": [{\"role\": \"user\", \"content\": \"I am hiring a mid-level software engineer and need cognitive ability and Java knowledge tests\"}]}"
```

## Docker

Ensure `catalog.json` is present, then:

```bash
docker build -t shl-recommender .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key_here shl-recommender
```

The image runs `build_index.py` during build so the container starts with a ready index.

## Deploy on Render

1. Push the repository to GitHub.
2. Create a **New Web Service** on [Render](https://render.com/).
3. Connect the repo and set:
   - **Environment variable:** `GEMINI_API_KEY` = your key (use Render’s secret UI, not the repo).
   - **Build command:** `pip install -r requirements.txt && python build_index.py`
   - **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Deploy. Render sets `$PORT` automatically.

## Evaluation

The project includes an explicit evaluation harness to measure retrieval quality, recommendation relevance, groundedness, schema compliance, and behavior probes (aligned with SHL grading criteria).

### Metrics

| Metric | What it measures |
|--------|------------------|
| **Retrieval Recall@K** | Fraction of labeled relevant assessments found in the top-K retrieval results (vector search + re-ranking). |
| **Agent Recall@K** | Same metric on the agent’s final `recommendations` after a full `/chat` run. |
| **Catalog validity rate** | Share of recommendation URLs that exist in `catalog.json`. |
| **Name groundedness rate** | Share of recommendations whose `name` exactly matches the canonical catalog name for that URL. |
| **Schema compliance rate** | Share of agent responses matching the required JSON schema. |
| **Behavior probe pass rate** | Binary checks (vague turn-1 → no recs, off-topic refuse, compare → no recs, refine, etc.). |

### Files

- `eval_cases.json` — labeled conversation traces, relevant catalog URLs, and behavior probes.
- `evaluate.py` — runs cases and prints a summary report.

### Run evaluation

Ensure the index is built first:

```bash
python build_index.py
```

**Retrieval only** (fast, no Gemini API calls):

```bash
python evaluate.py --retrieval-only
```

**Full evaluation** (retrieval + agent; requires `GEMINI_API_KEY` in `.env`):

```bash
python evaluate.py
```

Save a JSON report:

```bash
python evaluate.py --output eval_report.json
```

Optional: change K for Recall@K (default 10):

```bash
python evaluate.py --k 10 --output eval_report.json
```

### Adding cases

Edit `eval_cases.json`. For recommendation cases, set `relevant_urls` to ground-truth SHL catalog URLs. For behavior cases, set `probe` to one of:

- `no_recommendations_turn1_vague`
- `refuse_off_topic`
- `compare_no_recommendations`
- `has_recommendations`
- `refine_updates_shortlist`

## Project layout

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app (`/health`, `/chat`) |
| `agent.py` | Agent orchestration and validation |
| `retriever.py` | ChromaDB retrieval |
| `embedder.py` | Sentence-transformers embeddings |
| `catalog_loader.py` | Catalog load/validate |
| `prompts.py` | System prompts |
| `models.py` | Pydantic models |
| `build_index.py` | Index builder |
| `evaluate.py` | Evaluation harness and metrics |
| `eval_cases.json` | Labeled eval traces |
| `static/` | Web chat UI (HTML, CSS, JS) |
| `catalog.json` | SHL product catalog |

## Security

- Store `GEMINI_API_KEY` only in environment variables or a local `.env` file.
- `.env` is listed in `.gitignore` and must not be committed.
- Use `GEMINI_API_KEY` as the variable name (not custom names) so the app picks it up correctly.
- Never commit API keys or share `.env` in screenshots or chat.

## Gemini model

The agent calls **`gemini-1.5-flash`** first (per assignment spec). If that model is not available on your API project, it automatically falls back to `gemini-2.5-flash` and then `gemini-2.0-flash`. Ensure your API key has quota enabled in [Google AI Studio](https://aistudio.google.com/).
