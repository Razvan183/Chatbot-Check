# Chatbot Check

**A RAG quality-assurance MVP for testing whether a chatbot is good enough to
release.**

Most RAG demos stop once the chatbot answers a question. Chatbot Check focuses
on the next production problem: measuring whether a RAG system retrieves the
right evidence, answers from that evidence, cites sources, refuses unsupported
questions, and improves across versions.

The project includes a working FastAPI backend, SQLite persistence, a browser
demo, repeatable seed data, evaluation scorecards, regression comparison, and
CI-backed tests.

## Why It Matters

Teams shipping RAG systems need more than a chat endpoint. They need to answer:

- Did retrieval find the correct evidence?
- Is the answer grounded in that evidence?
- Are citations present and useful?
- Does the chatbot refuse questions the documents cannot support?
- Did a new prompt or runtime setting improve quality or create a regression?

Chatbot Check turns those questions into a small but complete QA workflow.

## What It Does

- Loads realistic company-policy documents from Markdown/text files.
- Splits documents into stable, searchable chunks.
- Retrieves relevant chunks with sentence-transformer embeddings, with an
  offline deterministic fallback.
- Builds strict citation-aware RAG prompts.
- Generates answers with deterministic mock mode or local Ollama.
- Logs chats, prompts, retrieved evidence, citations, latency, and settings.
- Seeds four chatbot versions for comparison.
- Runs a 50-case HR policy evaluation benchmark.
- Scores retrieval, citations, refusals, numeric consistency, and relevance.
- Produces single-run scorecards, tuning recommendations, comparisons, and
  printable HTML reports.
- Supports evaluating the internal demo RAG pipeline or an external HTTP RAG
  service through a connector.

## Tech Stack

| Area | Tools |
|---|---|
| API | FastAPI, Uvicorn |
| Data | SQLite, SQLAlchemy |
| Schemas | Pydantic |
| Retrieval | Sentence Transformers, NumPy cosine similarity |
| Evaluation | Custom deterministic metrics |
| Testing | Pytest, HTTPX |
| Deployment | Docker, GitHub Actions CI |

## Architecture

```text
Policy docs -> loader -> chunker -> SQLite
                                      |
User question -> retriever -> prompt -> generator -> answer + citations
                                      |
Evaluation dataset -> scoring -> scorecards + comparisons + reports
```

## Quick Start

### macOS/Linux

```bash
git clone <your-repository-url>
cd <repository-folder>
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python -m scripts.ingest_documents
python -m scripts.create_demo_versions
python -m scripts.import_eval_dataset
python -m uvicorn app.main:app --reload
```

### Windows PowerShell

```powershell
git clone <your-repository-url>
cd <repository-folder>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m scripts.ingest_documents
python -m scripts.create_demo_versions
python -m scripts.import_eval_dataset
python -m uvicorn app.main:app --reload
```

Open:

- Demo UI: <http://127.0.0.1:8000/>
- API docs: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>

The three seed scripts are required on a fresh clone. They create the local
`chatbot_check.db` file with documents, chunks, chatbot versions, and evaluation
cases. The database is intentionally ignored by Git.

## Demo Data

The local demo seeds:

- 7 company-policy documents
- 27 searchable chunks
- 4 chatbot configurations
- 50 evaluation cases

Seeded versions:

| Version | Purpose |
|---|---|
| `baseline_v1` | Default comparison baseline |
| `more_context_v2` | Retrieves more context |
| `strict_refusal_v3` | More deterministic/refusal-oriented behavior |
| `weak_bad_demo_v4` | Intentional regression example |

## Common Commands

Run the API:

```bash
python -m uvicorn app.main:app --reload
```

Run an evaluation:

```bash
python -m scripts.run_evaluation --version-name baseline_v1
```

Run tests:

```bash
python -m pytest
```

Build and run with Docker:

```bash
docker build -t chatbot-check .
docker run --rm -p 8000:8000 chatbot-check
```

## Main API Surface

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/chat` | Ask a question against a chatbot version |
| `GET` | `/chatbot-versions` | List seeded or tuned versions |
| `POST` | `/chatbot-versions` | Create a tuned runtime version |
| `GET` | `/documents` | List ingested documents |
| `GET` | `/evaluations/runs/{id}/scorecard` | View evaluation quality metrics |
| `GET` | `/evaluations/runs/{a}/compare/{b}` | Compare two evaluation runs |
| `GET` | `/evaluations/runs/{a}/compare/{b}/report` | Open a printable HTML report |
| `GET/POST` | `/rag-connectors` | Configure internal or external RAG evaluation |

## Testing and CI

The test suite is designed to run without external model or API calls. External
embedding/generation behavior is replaced with deterministic test doubles where
needed.

Current local result:

```text
140 passed
```

GitHub Actions runs `python -m pytest` on every push and pull request using
`.github/workflows/ci.yml`.

## Project Structure

```text
app/
|-- api/          FastAPI routes
|-- connectors/   Internal and HTTP RAG connectors
|-- db/           SQLAlchemy models, sessions, and Pydantic schemas
|-- evaluation/   Metrics, scoring, datasets, and run logic
|-- ingestion/    Document loading, chunking, and embeddings
|-- rag/          Retrieval, prompts, generation, and chat pipeline
`-- main.py       FastAPI entry point

data/             Demo documents and evaluation dataset
demo/             Browser UI
scripts/          Seed and evaluation scripts
tests/            Automated test suite
```

## Engineering Decisions

- **Simple infrastructure:** SQLite keeps the MVP easy to run and inspect.
- **Repeatable seeding:** documents, versions, and eval cases can be recreated
  from scripts.
- **Deterministic testing:** tests avoid network/model dependency.
- **Traceable evaluation:** runs store settings, retrieved evidence, metrics,
  and failure labels.
- **Connector boundary:** evaluations can target the built-in RAG pipeline or
  an external HTTP RAG system.

## Current Limitations

- Chunk embeddings are recomputed during retrieval instead of stored in a vector
  index.
- SQLite table creation is used instead of Alembic migrations.
- Evaluation jobs use FastAPI background tasks, not a durable worker queue.
- The browser UI is a lightweight demo, not a polished analytics dashboard.
- Authentication, pagination, observability, and production deployment hardening
  are intentionally out of scope for this MVP.

## What This Demonstrates

- End-to-end RAG application design
- Practical RAG evaluation and regression testing
- Backend API development with FastAPI
- Relational modeling for AI system traces and evaluations
- Testable Python with clear module boundaries
- Product thinking around reliability, grounding, and release readiness

## License

This project is currently intended for educational and portfolio use.
