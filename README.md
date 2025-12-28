# RAG Project

Overview
--------
This repository implements a Retrieval-Augmented Generation (RAG) pipeline for TED Talks transcripts. It builds an embedding index (Pinecone) from the CSV data and runs contextualized queries through a chat-style LLM. A small FastAPI server exposes the RAG pipeline as HTTP endpoints for local use or deployment (e.g., Vercel).

Repository structure
- `create_index.py` — build and upsert chunk embeddings into Pinecone.
- `query_index.py` — retrieve relevant chunks and run the RAG QA pipeline.
- `Constants.py` — project configuration (chunk sizes, batch sizes, paths).
- `system_prompt.text` — system prompt used when calling the chat model.
- `data/` — source CSV files and sample data:
	- `data/ted_talks_en.csv` — full TED Talks dataset (CSV).
	- `data/ted_talks_en-small.csv` — smaller sample CSV for quick tests.
- `api/` — FastAPI application and related files:
	- `api/prompt.py` — FastAPI app exposing the RAG endpoints (`POST /api/prompt`, `GET /api/stats`).
	- `api/stats.json` — local stats/config output used by the API (if present).
- `vercel.json` — Vercel deployment configuration (if you deploy the API to Vercel).

Prerequisites (index creation)
if you want to create an index using this code, you will need:
- Python 3.8+ recommended.
- A Pinecone account and API key (if using Pinecone index storage).
- an index created in your account

Prerequisites (using RAG)
If you want to apply retrival augmented genration you will need:
- An index in Pinecone (created in the index creation part)
- An LLM provider (the project uses a `langchain-openai` wrapper configured with `LLMOD_API_KEY` and `LLMOD_URL`).

Prerequisites (connecting project to vercel)
- vercel account
- environment variables defining the previous Prerequisites

Prerequisites (using application via this projects vercel)
- link to project in vercel only, the rest is handled by code and vercel

Install
-------
Create and activate a virtual environment, then install dependencies from `requirements.txt`:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Environment variables
---------------------
Create a `.env.local` file in the project root with these keys (replace placeholders):

```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=ted-talks
PINECONE_INDEX_SMALL=ted-talks-small
LLMOD_API_KEY=your_llm_provider_api_key
LLMOD_URL=https://your-llm-endpoint.example.com
EMBEDDING_MODEL=embedding-model-name
LLM_MODEL=chat-model-name
```

If you plan to deploy to Vercel or call the Vercel API, create a `VERCEL_TOKEN` on the Vercel dashboard and keep it secret.

Usage
-----
1) Build the index locally (optional, required for search against Pinecone):

```bash
python create_index.py
```

2) Query locally (script):

```bash
python query_index.py
```

Web API (FastAPI)
------------------
This project exposes two HTTP endpoints defined in `api/prompt.py`:

- `POST /api/prompt`
	- Request JSON: `{ "question": "your question here" }`
	- Returns: the RAG pipeline response (chat + retrieved context).

- `GET /api/stats`
	- Returns basic config values used by the pipeline (chunk size, overlap ratio, top_k).

Run the FastAPI app locally using `uvicorn`:

```bash
# serve on default <localhost> (replace with your local host)
uvicorn api.prompt:app --reload

# or specify host/port explicitly
uvicorn api.prompt:app --reload --host <localhost> --port 8000
```

Example requests
----------------
Using `curl`:

```bash
# POST prompt
curl -X POST "http://<localhost>:8000/api/prompt" \
	-H "Content-Type: application/json" \
	-d '{"question":"Find a TED talk that discusses how to succeed. Provide the title and speaker."}'

# GET stats
curl "http://<localhost>:8000/api/stats"
```

PowerShell (Invoke-RestMethod):

```powershell
# POST
Invoke-RestMethod -Method POST -Uri "http://<localhost>:8000/api/prompt" -ContentType "application/json" -Body '{"question":"Find a TED talk that discusses how to succeed. Provide the title and speaker."}'

# GET
Invoke-RestMethod -Method GET -Uri "http://<localhost>1:8000/api/stats"
```

Python (requests):

```python
import requests
resp = requests.post(
		"http://<localhost>:8000/api/prompt",
		json={"question": "Find a TED talk that discusses how to succeed. Provide the title and speaker."}
)
print(resp.json())
```

Security
--------
- Keep all API tokens and keys out of the repo. Use a `.env.local` file or secret manager.
- Do not expose admin tokens publicly. When deploying, use Vercel/host provider environment secrets.

Expected outputs
----------------
- After `create_index.py`, your Pinecone index will contain vector records.
- The API `POST /api/prompt` returns the RAG response JSON, and `GET /api/stats` returns pipeline config values.


More options implementted in the code:
--------------------------------------

Small-index option
-------------------
This repo supports creating and using a smaller test index for fast iteration without rebuilding the full production index.

- What it does:
	- `create_small_csv(path_to_big_csv, path_to_small_csv, num_rows)` — extracts the first `num_rows` rows from the main CSV to a small CSV for testing.
	- Chunking functions (`chunk_one_script`, `create_all_chunks`) split transcripts into overlapping chunks used as vector records.
	- Embedding and upsert are handled by `embed_one_batch` and `add_chunks_to_index_batch`, respectively.

- Quick examples (PowerShell):

```powershell
# create a small CSV with 100 rows
python -c "from create_index import create_small_csv; create_small_csv('data/ted_talks_en.csv','data/ted_talks_en-small.csv',100)"

# build and upsert to the small index (option A: set env var then run)
$env:PINECONE_INDEX = 'ted-talks-small'
python create_index.py

# build directly with helper (option B)
python -c "from create_index import run_full_pipeline_batch, PINECONE_INDEX_SMALL; run_full_pipeline_batch('data/ted_talks_en-small.csv', PINECONE_INDEX_SMALL, batch_size=10)"
```

Notes:
	- Use the small-index option for quicker iterations and debugging. Keep the small index in a separate Pinecone index name to avoid conflicts with the full dataset.
	- Adjust `batch_size` for your environment to avoid memory spikes.

Retrieval & RAG pipeline
------------------------
The retrieval pipeline is implemented in `query_index.py`. It provides functions that can be used programmatically or via the FastAPI endpoints.

- Main steps:
	1. `query_index(index, embeddings_model, query_text, top_k)` — embed the query and query Pinecone for top matches.
	2. `join_query_results(matches)` — group matches by `talk_id` and collect per-talk metadata and chunk texts.
	3. `build_context_from_matches(talk_chunks, max_chunks_per_talk)` — sort chunks by score and build a text context limited by `MAX_CHUNKS_PER_TALK`.
	4. `run_query_in_model(context, system_prompt_path, user_query, chat_model)` — loads `system_prompt.text`, builds the system+user messages and calls the chat model.
	5. `format_rag_response(...)` and `full_query_pipeline(...)` — assemble the final JSON output.

- Usage:

```bash
# Run all JSON queries in `queries/` and save outputs to `results/`
python query_index.py

# Or run the FastAPI server and POST a single question
uvicorn api.prompt:app --reload --port 8000
curl -X POST "http://127.0.0.1:8000/api/prompt" -H "Content-Type: application/json" -d '{"question":"Your question"}'
```

Notes:
	- `query_index.py` includes `full_query_pipeline` and `multiple_queries_pipeline`; `main()` runs the multiple-queries pipeline and writes outputs to `results/`.
	- For single-query automation prefer calling `full_query_pipeline(...)` directly or using the FastAPI `POST /api/prompt` endpoint.


