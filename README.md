# RAG Project

Overview
--------
This repository implements a simple Retrieval-Augmented Generation (RAG) pipeline for TED Talks transcripts. It builds an embedding index (Pinecone) from the CSV data and runs contextualized queries through a chat-style LLM.

Repository structure
- `create_index.py` — build and upsert chunk embeddings into Pinecone.
- `query_index.py` — retrieve relevant chunks and run the RAG QA pipeline.
- `Constants.py` — project configuration (chunk sizes, batch sizes, paths).
- `data/` — source CSV files (`ted_talks_en.csv`, `ted_talks_en-small.csv`).
- `queries/` — example query JSON files.
- `results/` — output JSON files produced by the query pipeline.
- `system_prompt.text` — system prompt used when calling the chat model.

Prerequisites
- Python 3.8+ recommended.
- A Pinecone account and API key.
- An LLM provider compatible with the project's `langchain_openai` wrapper (set with `LLMOD_API_KEY` and `LLMOD_URL`).
- Install required Python packages (example command below).

Environment variables
Create a `.env.local` file in the project root with the following keys (replace placeholders with your values):

PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=ted-talks
PINECONE_INDEX_SMALL=ted-talks-small
LLMOD_API_KEY=your_llm_provider_api_key
LLMOD_URL=https://your-llm-endpoint.example.com
EMBEDDING_MODEL=embedding-model-name
LLM_MODEL=chat-model-name

Setup
-----
1. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Install the typical dependencies used by this project:

```bash
pip install pandas python-dotenv tqdm pinecone-client openai langchain
```

(The project has no `requirements.txt`; adjust packages as needed if your environment differs.)

Usage
-----
1. Build the index (embeddings + upsert). This reads `data/ted_talks_en.csv` by default and upserts vectors to the Pinecone index configured in `.env.local`:

```bash
python create_index.py
```

2. Run queries against the index. Results are saved to the `results/` folder:

```bash
python query_index.py
```

Notes & tips
- For quick tests, use the smaller CSV at `data/ted_talks_en-small.csv` and set `PINECONE_INDEX_SMALL` in `.env.local`.
- The code expects these constant values in `Constants.py` (chunk size, batch sizes, TOP_K, etc.). Adjust them if you need different behavior.
- Keep secrets out of version control. Do not commit `.env.local`.

Expected outputs
- After `create_index.py` completes, your Pinecone index will contain vector records (IDs like `talkid_chunk_N`).
- After `query_index.py` completes, JSON outputs appear in `results/` (one per input query JSON in `queries/`).

Next steps
- Want me to run a local smoke test (build the small index) or create a `requirements.txt`? Reply with which you'd prefer.

