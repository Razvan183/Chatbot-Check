# ChatbotCheck

**An in-progress quality-assurance platform for retrieval-augmented generation
(RAG) systems.**

ChatbotCheck is designed to evaluate whether chatbot answers are grounded in
retrieved documents, correctly cited, safe to release, and improving across
different RAG configurations.

The current milestone implements the complete document ingestion and semantic
retrieval foundation: source documents are loaded, split into overlapping
chunks, persisted in SQLite, exposed through FastAPI, and ranked against user
questions with sentence-transformer embeddings.

> **Current status:** ingestion and retrieval are working and covered by 50
> automated tests. Answer generation and the evaluation engine are the next
> development stages.

## Why this project?

Building a RAG chatbot is only the first step. A production team must also be
able to answer:

- Did the retriever find the correct evidence?
- Is the answer supported by that evidence?
- Does the chatbot refuse questions that cannot be answered safely?
- Did a new prompt or configuration improve the system—or cause a regression?

ChatbotCheck is being built to make these questions measurable and repeatable.

## Current capabilities

- FastAPI backend with interactive OpenAPI documentation
- Environment-based application configuration
- SQLite persistence through SQLAlchemy
- Eight-model database schema for documents, chatbot versions, and evaluations
- Seven realistic company-policy source documents
- Deterministic UTF-8 Markdown and text loading
- Paragraph-aware chunking with configurable, size-safe overlap
- Atomic and repeatable document ingestion
- Lazy, cached sentence-transformer model loading
- Cosine-similarity ranking of stored document chunks
- Four reproducible chatbot configurations for future comparison
- Read-only document and chunk API endpoints
- Isolated unit and integration-style tests

## Architecture

```text
Policy documents
       │
       ▼
Document loader
       │
       ▼
Paragraph-aware chunker
       │
       ▼
SQLAlchemy ingestion ───────────────► SQLite
                                         │
User question                            │
       │                                 │
       ▼                                 ▼
Sentence-transformer embeddings ◄── Stored chunks
       │
       ▼
Cosine-similarity ranking
       │
       ▼
Top-k relevant chunks
```

FastAPI provides access to the stored documents, while the retrieval layer can
be called from Python. A later milestone will connect retrieval to prompt
construction, answer generation, and automated evaluation.

## Technology stack

| Area | Technology |
|---|---|
| API | FastAPI, Uvicorn |
| Validation | Pydantic |
| Persistence | SQLAlchemy, SQLite |
| Embeddings | Sentence Transformers |
| Vector operations | NumPy |
| Testing | Pytest, HTTPX |
| Configuration | python-dotenv |

## Project structure

```text
app/
├── api/             Document API endpoints
├── db/              Database connection, models, and schemas
├── ingestion/       File loading, chunking, and embeddings
├── rag/             Semantic retrieval
├── config.py        Environment-based settings
└── main.py          FastAPI entry point

data/
└── sample_company_policy/   Demo policy corpus

scripts/
├── inspect_documents.py     Inspect source files
├── ingest_documents.py      Load and persist the corpus
└── create_demo_versions.py  Seed RAG configurations

tests/                        Automated test suite
```


## Getting started

### Prerequisites

- Python 3.10 or newer
- Git

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <repository-folder>
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the application

Configuration has local defaults, so this step is optional.

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

Available settings include the database URL, data directory, chunk size,
overlap, embedding model, and mock-LLM mode.

### 5. Ingest the demo documents

```bash
python -m scripts.ingest_documents
```

The current sample corpus produces:

```text
Loaded 7 documents
Created 27 chunks
Saved documents and chunks to database
```

The operation replaces previously ingested demo documents in one transaction,
so it is safe to run repeatedly.

### 6. Seed the chatbot configurations

```bash
python -m scripts.create_demo_versions
```

This creates or updates four configurations:

| Version | Top-k | Temperature | Purpose |
|---|---:|---:|---|
| `baseline_v1` | 3 | 0.2 | Baseline configuration |
| `more_context_v2` | 5 | 0.2 | Retrieves additional context |
| `strict_refusal_v3` | 5 | 0.0 | More deterministic behavior |
| `weak_bad_demo_v4` | 1 | 0.7 | Intentional regression example |

### 7. Start the API

```bash
python -m uvicorn app.main:app --reload
```

Open:

- Health check: <http://127.0.0.1:8000/health>
- Interactive API documentation: <http://127.0.0.1:8000/docs>
- Documents: <http://127.0.0.1:8000/documents>

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Check whether the API is running |
| `GET` | `/documents` | List ingested documents |
| `GET` | `/documents/{document_id}` | Get one document |
| `GET` | `/documents/{document_id}/chunks` | List its chunks in source order |

Example health response:

```json
{
  "status": "ok",
  "service": "evalforge-api"
}
```

## Semantic retrieval

The retrieval layer:

1. reads all stored chunks and their document metadata;
2. embeds the question and chunks in one batch;
3. calculates cosine similarity for every chunk;
4. sorts by descending score with a deterministic tie-breaker;
5. returns the requested number of results.

Example:

```python
from app.rag.retriever import retrieve_chunks

results = retrieve_chunks(
    "How many vacation days do employees receive after two years?",
    top_k=3,
)

for result in results:
    print(result["score"], result["filename"])
    print(result["chunk_text"])
```

The first real retrieval may download
`sentence-transformers/all-MiniLM-L6-v2` if it is not already cached.

## Testing

Run the complete suite:

```bash
python -m pytest
```

Current result:

```text
50 passed
```

The tests cover:

- loader validation and deterministic ordering;
- chunk-size limits, overlap, headings, and content preservation;
- embedding input validation, model caching, and cosine similarity;
- repeatable transactional ingestion;
- semantic ranking and edge cases;
- database foreign-key enforcement;
- API ordering, serialization, and 404 responses;
- idempotent chatbot-version seeding.

External embedding calls are replaced with deterministic test doubles, keeping
the suite fast and independent of network availability.

## Notable engineering decisions

- **Simple before distributed:** SQLite and in-memory ranking keep the current
  implementation transparent and easy to inspect.
- **Atomic ingestion:** old and new corpus records are replaced in one
  transaction, with rollback on failure.
- **Bounded overlap:** chunks receive prior context only when it fits inside the
  configured size limit.
- **Lazy model initialization:** the embedding model is loaded on first use and
  cached for the process lifetime.
- **Dependency injection:** database session factories can be replaced with
  isolated in-memory databases during tests.
- **Deterministic behavior:** files, API responses, chunks, and equal-score
  retrieval results have explicit ordering.

## Current limitations

- Retrieval is not yet exposed as an HTTP endpoint.
- The project does not yet generate chatbot answers.
- Chatbot-version settings are stored but not yet connected to runtime behavior.
- Chunk embeddings are recomputed for each retrieval request.
- SQLite table creation is used instead of a migration framework.
- Authentication, pagination, observability, and production deployment are not
  implemented yet.

These choices are intentional for the current learning-focused MVP stage.

## Roadmap

- [x] FastAPI application and configuration
- [x] SQLAlchemy database schema
- [x] Sample company-policy corpus
- [x] Document loading and chunking
- [x] Repeatable database ingestion
- [x] Document API endpoints
- [x] Embeddings and semantic retrieval
- [x] Reproducible chatbot configurations
- [ ] RAG prompt and answer generator
- [ ] End-to-end chat pipeline and API
- [ ] Evaluation datasets and metrics
- [ ] Failure classification and regression comparison
- [ ] Dashboard and HTML reports
- [ ] Docker and CI quality gates

## What this project demonstrates

- Designing a modular RAG backend from first principles
- Building deterministic ingestion and retrieval pipelines
- Modeling future evaluation workflows relationally
- Writing testable Python through clear boundaries and dependency injection
- Handling edge cases instead of relying only on happy-path demos
- Developing an AI system incrementally while keeping completed and planned
  functionality clearly separated


## License

This project is currently intended for educational and portfolio use.
