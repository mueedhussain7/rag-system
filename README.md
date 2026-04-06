# RAG System — Auto-Updating, Hallucination-Aware RAG with Evaluation Dashboard

A production-level Retrieval-Augmented Generation (RAG) system that automatically scores every answer for hallucination risk and tracks quality metrics on a live evaluation dashboard.

## What this project does

Most RAG systems retrieve documents and generate answers — but have no way of knowing whether the answer is actually grounded in the retrieved content. This project adds a **hallucination detection layer** on top of a full RAG pipeline, automatically scoring every answer for faithfulness before it reaches the user.

```
PDF / URL / .txt
      ↓
Document ingestion (chunk → embed → store in ChromaDB)
      ↓
Hybrid retrieval (semantic search + BM25 + RRF fusion)
      ↓
Grounded answer generation (GPT-4o-mini + streaming)
      ↓
Hallucination scoring (RAGAS faithfulness + NLI checker)
      ↓
Logged to SQLite → visible on Streamlit evaluation dashboard
```

---

## Key features

- **Hybrid search** — combines semantic vector search with BM25 keyword search using Reciprocal Rank Fusion, consistently outperforming either method alone
- **Hallucination detection** — RAGAS faithfulness scoring (0–1) and sentence-level NLI verification on every answer
- **Auto-updating** — APScheduler background job re-ingests documents every 24 hours, keeping the knowledge base fresh without manual intervention
- **Streaming responses** — answers stream word-by-word via FastAPI `StreamingResponse`
- **Source citations** — every answer includes the document and page number it drew from
- **Evaluation dashboard** — Streamlit UI showing faithfulness trends, hallucination rates, latency distributions, and full query history
- **SQLite logging** — every query, answer, score, and ingestion event is persisted for analysis

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| LLM & embeddings | OpenAI GPT-4o-mini + text-embedding-3-small |
| Orchestration | LangChain (LCEL chains, document loaders, text splitters) |
| Vector database | ChromaDB (local, persistent) |
| Keyword search | rank-bm25 |
| Hallucination eval | RAGAS 0.4.3 |
| Auto-updating | APScheduler |
| Logging | SQLite + SQLAlchemy |
| Dashboard | Streamlit + Plotly |
| Testing | pytest |

---

## Project structure

```
rag-system/
├── app/
│   ├── main.py                  # FastAPI app — all endpoints
│   ├── config.py                # Central config via pydantic-settings
│   ├── ingestion/
│   │   ├── loaders.py           # PDF, txt, URL document loaders
│   │   ├── chunker.py           # RecursiveCharacterTextSplitter
│   │   └── embedder.py          # OpenAI embeddings + ChromaDB writer
│   ├── retrieval/
│   │   ├── semantic.py          # ChromaDB vector similarity search
│   │   ├── keyword.py           # BM25 keyword search
│   │   ├── hybrid.py            # RRF fusion + reranking
│   │   └── context.py           # Context assembler for prompts
│   ├── generation/
│   │   ├── chain.py             # LangChain RAG chain
│   │   ├── prompt.py            # Grounded prompt template
│   │   └── scheduler.py         # APScheduler auto-refresh
│   ├── hallucination/
│   │   ├── ragas_scorer.py      # RAGAS faithfulness scoring
│   │   ├── nli_checker.py       # Sentence-level NLI via GPT-4o-mini
│   │   └── scorer.py            # Combined hallucination scorer
│   └── evaluation/
│       ├── logger.py            # SQLite query + ingestion logger
│       └── dashboard.py         # Streamlit evaluation dashboard
├── data/
│   └── documents/               # Place your source documents here
├── tests/
│   ├── test_health.py
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   ├── test_hallucination.py
│   └── test_evaluation.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## Getting started

### Prerequisites

- Python 3.13+
- An OpenAI API key with credits ([platform.openai.com](https://platform.openai.com))

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/mueedhussain7/rag-system.git
cd rag-system

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Open .env and add your OpenAI API key
```

### Running the API

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.  
Interactive docs at `http://localhost:8000/docs`.

### Running the dashboard

In a second terminal (with venv active):

```bash
streamlit run app/evaluation/dashboard.py
```

The dashboard opens at `http://localhost:8501`.

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server health check |
| `POST` | `/ingest` | Ingest a document (PDF, txt, or URL) |
| `GET` | `/retrieve` | Retrieve top-k chunks for a query |
| `POST` | `/ask` | Generate a grounded, cited, scored answer |
| `POST` | `/ask/stream` | Same as `/ask` but streams tokens |
| `POST` | `/score` | Score any answer for hallucination risk |
| `GET` | `/metrics` | Current system health metrics as JSON |

### Example: ingest a document

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source": "data/documents/my_paper.pdf"}'
```

### Example: ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"source": "What are the main findings of the study?"}'
```

Response:

```json
{
  "question": "What are the main findings of the study?",
  "answer": "The study found that all participants used mobile devices...",
  "sources": ["data/documents/my_paper.pdf (page 8)"],
  "chunks_used": 5,
  "faithfulness_score": 0.95,
  "confidence_level": "high",
  "nli_verdict": "clean",
  "latency_ms": 4821.3
}
```

---

## Hallucination scoring explained

Every answer produced by `/ask` is automatically scored before being returned.

| Field | What it means |
|---|---|
| `faithfulness_score` | RAGAS score (0–1). How much of the answer is supported by the retrieved context. |
| `confidence_level` | `high` (≥0.8, clean) · `medium` (≥0.5) · `low` (<0.5 or contradicted) · `unverified` (scoring failed) |
| `nli_verdict` | `clean` · `uncertain` · `contradicted` — based on sentence-level NLI classification |

---

## Running tests

```bash
pytest tests/ -v
```

All 29 tests should pass with 0 warnings.

---

## Environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
OPENAI_API_KEY=sk-your-key-here
EMBEDDING_MODEL=text-embedding-3-small
CHROMA_DB_PATH=./chroma_db
CHROMA_COLLECTION_NAME=rag_documents
CHUNK_SIZE=500
CHUNK_OVERLAP=50
APP_ENV=development
APP_VERSION=0.1.0
LOG_LEVEL=INFO
USER_AGENT=rag-system/0.1.0
```

## Design decisions

**Why ChromaDB over Pinecone?** ChromaDB runs locally with zero setup and no API costs, making it ideal for development and portfolio projects. The interface is compatible with hosted solutions if you want to scale.

**Why BM25 + semantic search?** Neither method alone handles all query types well. Semantic search misses exact keyword matches (product names, codes); BM25 misses paraphrased queries. Hybrid search with RRF consistently outperforms either individually.

**Why RAGAS over manual evaluation?** RAGAS provides automated, reproducible faithfulness scoring using LLMs as evaluators — the same approach used by teams at production AI companies. It's fast enough to run on every query.

**Why `temperature=0` for generation?** Zero temperature makes GPT-4 deterministic and less likely to hallucinate. Slightly higher temperature is used for self-consistency checks in the NLI pipeline.

---

## License

MIT — see [LICENSE](LICENSE) for details.
