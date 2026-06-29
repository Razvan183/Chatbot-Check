# Chatbot Check

**An in-progress quality-assurance platform for retrieval-augmented generation
(RAG) systems.**

Chatbot Check is designed to score whether a configured RAG system retrieves the
right evidence, answers from that evidence, cites sources, refuses unsupported
questions, and gives practical tuning guidance when quality drops.

The current milestone implements a working local RAG backend: source documents
are loaded, split into overlapping chunks, persisted in SQLite, retrieved with
sentence-transformer embeddings, passed into a strict citation-aware prompt, and
answered through either deterministic mock generation or a local Ollama model.

> **Current status:** ingestion, retrieval, prompt construction, answer
> generation, chat logging, starter evaluation metrics, queued evaluation runs,
> single-run scorecards, tuning recommendations, runtime configuration creation,
> run comparison, printable HTML reports, and the document/chat/evaluation APIs
> are working and covered by the automated test suite. Production deployment
> hardening is the next major development stage.

## Why this project?

Building a RAG chatbot is only the first step. A production team must also be
able to answer:

- Did the retriever find the correct evidence?
- Is the answer supported by that evidence?
- Does the chatbot refuse questions that cannot be answered safely?
- Which runtime settings should we change when a score is weak?
- Did a new prompt or configuration improve the system, or cause a regression?

Chatbot Check is being built to make these questions measurable and repeatable.

## Current capabilities

- FastAPI backend with interactive OpenAPI documentation
- Environment-based application configuration
- SQLite persistence through SQLAlchemy
- Eight-model database schema for documents, chatbot versions, and evaluations
- Seven realistic company-policy source documents
- Deterministic UTF-8 Markdown and text loading
- Paragraph-aware chunking with configurable, size-safe overlap
- Atomic and repeatable document ingestion
- Stable chunk keys for evidence tracking across database resets
- Lazy, cached sentence-transformer model loading
- Cosine-similarity ranking of stored document chunks
- Strict RAG prompt construction with citation and refusal rules
- Mock answer generation with optional local Ollama support
- End-to-end RAG pipeline with chat logging
- Four reproducible chatbot configurations for future comparison
- 50-case HR policy evaluation benchmark
- Custom evaluation metrics for citations, refusals, retrieval hits, numbers,
  and keyword overlap
- Evaluation runner that saves case scores and failure labels
- Queued evaluation API endpoints for launching and reviewing runs
- Single-run RAG quality scorecards with metric breakdowns
- Tuning recommendations for weak retrieval, citations, refusals, and numbers
- UI workflow for creating tuned runtime versions
- Connector interface for evaluating internal or external RAG systems
- Persisted RAG connector configuration and test endpoint
- LLM-generated draft evaluation datasets with human approval gates
- Regression comparison API for two evaluation runs
- Printable HTML comparison reports for sharing release decisions
- Chat, document, chunk, version, and evaluation API endpoints
- Browser demo for chat, evidence inspection, documents, evaluations, and reports
- Isolated unit and integration-style tests

## Architecture

```text
Policy documents
       |
       v
Document loader
       |
       v
Paragraph-aware chunker
       |
       v
SQLAlchemy ingestion ------------------> SQLite
                                         |
User question                            |
       |                                 |
       v                                 v
Sentence-transformer embeddings <--- Stored chunks
       |
       v
Cosine-similarity ranking
       |
       v
Top-k relevant chunks
       |
       v
Strict RAG prompt
       |
       v
Mock or local Ollama answer generator
       |
       v
Chat response and chat log
```

FastAPI provides access to stored documents and to the chat pipeline. A later
milestone will add automated evaluation datasets, scoring, failure
classification, and version comparison.

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
|-- api/             Chat and document API endpoints
|-- connectors/      RAG target connector interface and implementations
|-- db/              Database connection, models, and schemas
|-- ingestion/       File loading, chunking, and embeddings
|-- rag/             Retrieval, prompts, generation, and pipeline logic
|-- config.py        Environment-based settings
`-- main.py          FastAPI entry point

data/
`-- sample_company_policy/   Demo policy corpus

scripts/
|-- inspect_documents.py     Inspect source files
|-- ingest_documents.py      Load and persist the corpus
|-- create_demo_versions.py  Seed RAG configurations
|-- import_eval_dataset.py   Seed the HR policy evaluation benchmark
`-- run_evaluation.py        Run a saved dataset against a chatbot version

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

Available settings include the app name, service name, database URL, data
directory, chunk size, overlap, embedding model, mock-LLM mode, and local
Ollama settings. Evaluation can target the built-in RAG pipeline or an external
HTTP RAG endpoint through `RAG_CONNECTOR`. Dataset generation has its own
provider settings so the authoring LLM can be stronger and separate from the
RAG model under test.

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

### 7. Import the evaluation dataset

```bash
python -m scripts.import_eval_dataset
```

This creates or replaces the 50-case `HR Policy Eval Set`.

### 8. Run an evaluation

```bash
python -m scripts.run_evaluation --version-name baseline_v1
```

The runner creates an evaluation run, asks every dataset question through the
chat pipeline, stores per-case metrics, classifies failures, and prints a
summary.

### 9. Start the API

```bash
python -m uvicorn app.main:app --reload
```

Open:

- Demo UI: <http://127.0.0.1:8000/>
- Health check: <http://127.0.0.1:8000/health>
- Interactive API documentation: <http://127.0.0.1:8000/docs>
- Documents: <http://127.0.0.1:8000/documents>

The demo UI provides a browser-based workspace for asking questions, switching
between chatbot versions, inspecting retrieved evidence, creating tuned runtime
versions, browsing the ingested corpus, and launching or reviewing evaluation
runs.

### Docker deployment

Build a demo-ready container image:

```bash
docker build -t chatbot-check .
```

Run it locally:

```bash
docker run --rm -p 8000:8000 chatbot-check
```

The image seeds the sample policy corpus, chatbot versions, and 50-case
evaluation dataset during build. For persistent deployments, set
`DATABASE_URL` to a mounted SQLite path or an external database URL supported by
SQLAlchemy.

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Check whether the API is running |
| `POST` | `/chat` | Ask a question through the RAG chatbot |
| `GET` | `/chatbot-versions` | List available chatbot configurations |
| `POST` | `/chatbot-versions` | Create a tuned runtime chatbot configuration |
| `GET` | `/documents` | List ingested documents |
| `GET` | `/documents/{document_id}` | Get one document |
| `GET` | `/documents/{document_id}/chunks` | List its chunks in source order |
| `POST` | `/evaluation-drafts` | Generate a draft evaluation dataset from ingested chunks |
| `GET` | `/evaluation-drafts` | List generated draft datasets |
| `GET` | `/evaluation-drafts/{draft_dataset_id}` | Review draft cases with supporting source chunks |
| `PATCH` | `/evaluation-drafts/{draft_dataset_id}/cases/{draft_case_id}` | Approve, reject, or edit one draft case |
| `POST` | `/evaluation-drafts/{draft_dataset_id}/publish` | Publish approved draft cases as an official dataset |
| `GET` | `/evaluations/datasets` | List evaluation datasets |
| `GET` | `/evaluations/runs` | List evaluation runs |
| `POST` | `/evaluations/runs` | Queue one dataset run against one chatbot version |
| `GET` | `/evaluations/runs/{eval_run_id}` | Get one evaluation run summary |
| `GET` | `/evaluations/runs/{eval_run_id}/scorecard` | Get one run's RAG quality scorecard and tuning recommendations |
| `GET` | `/evaluations/runs/{eval_run_id}/results` | List case-level run results |
| `GET` | `/evaluations/runs/{baseline_run_id}/compare/{candidate_run_id}` | Compare two runs from the same dataset |
| `GET` | `/evaluations/runs/{baseline_run_id}/compare/{candidate_run_id}/report` | Open a printable HTML comparison report |

Example health response:

```json
{
  "status": "ok",
  "service": "chatbot-check-api"
}
```

## Chat API

Ask a question against a seeded chatbot version:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"When should lost equipment be reported?\",\"chatbot_version_id\":1}"
```

The response includes the generated answer, retrieved chunks, extracted
citations, and request latency.

## Evaluation

The evaluation layer currently supports:

1. importing the HR policy benchmark dataset from JSON;
2. calculating deterministic metrics for each answer;
3. combining metrics into an overall case score;
4. assigning a simple failure type such as `missing_citation`,
   `wrong_number`, or `answered_unanswerable_question`;
5. scoring retrieval against stable chunk keys when the dataset provides them;
6. persisting `EvalRun` and `EvalResult` rows for later comparison and reports;
7. aggregating single-run scorecards for retrieval, citation, refusal, numeric
   consistency, and answer relevance;
8. recommending tuning changes such as increasing `top_k`, lowering
   `temperature`, tightening prompts, adding evidence thresholds, or adding
   reranking/hybrid retrieval;
9. comparing two runs side by side and rendering a standalone HTML report.

The bundled dataset includes 50 cases across factual lookups, citation-required
answers, numeric consistency checks, misleading questions, unanswerable
questions, and multi-document policy scenarios.

The report endpoint renders the same comparison data as standalone HTML. It is
intended for release reviews, stakeholder sign-off, and saving or printing a
version comparison without requiring someone to use the demo UI.

## Tuning workflow

The primary workflow is:

```text
Choose a configured RAG version
Configure or test the target RAG connector
Run the evaluation dataset
Review the quality scorecard
Create a tuned runtime version
Run the evaluation again
Optionally compare the two runs
```

The UI currently exposes query-time settings that can be changed without
re-ingesting documents:

| Setting | Why change it? |
|---|---|
| `top_k` | Increase when retrieval misses supporting evidence; decrease when context is noisy. |
| `temperature` | Lower when answers invent numbers or drift from evidence. |
| `model_name` | Switch between mock generation and a local Ollama model. |
| `prompt_template` | Tighten citation, refusal, and grounding behavior. |

Index-time settings such as `chunk_size`, `chunk_overlap`, and
`embedding_model` are stored on versions for traceability, but changing them
properly requires re-ingesting or rebuilding the document index.

## RAG connectors

Evaluation runs call a connector instead of being hard-wired to one RAG
implementation. The default connector is:

```env
RAG_CONNECTOR=internal
```

This evaluates Chatbot Check's built-in demo RAG pipeline. To evaluate an
external RAG service, configure:

```env
RAG_CONNECTOR=http
HTTP_RAG_URL=http://localhost:9000/chat
HTTP_RAG_TIMEOUT_SECONDS=60
```

The HTTP connector sends:

```json
{
  "question": "What is the policy?",
  "chatbot_version_id": 1
}
```

It expects a JSON response with an answer and, optionally, retrieved contexts:

```json
{
  "answer": "Employees receive 21 vacation days. [4]",
  "retrieved_chunks": [
    {
      "chunk_id": 4,
      "chunk_key": "vacation_policy.md::0",
      "chunk_text": "Full-time employees receive 21 paid vacation days..."
    }
  ],
  "citations": [4],
  "latency_ms": 120
}
```

Common alternatives such as `response`, `text`, `contexts`, and `sources` are
also normalized. This is the first step toward using Chatbot Check as a wrapper
around arbitrary RAG systems.

The demo UI includes a **Connect RAG** panel for choosing `internal` or `http`,
saving the target URL/timeout, and testing the connector. Saved connector
settings are persisted in SQLite and used by future evaluation runs.

## Dataset authoring LLM

Dataset authoring is separated from RAG answering. The target RAG may use a
local model through Ollama or an external HTTP endpoint, while candidate eval
case generation can use a stronger API-backed model.

By default, autonomous dataset generation is disabled:

```env
DATASET_GENERATOR_MODE=disabled
```

To enable Gemini-backed dataset authoring:

```env
DATASET_GENERATOR_MODE=gemini
DATASET_GENERATOR_MODEL=gemini-2.5-flash
DATASET_GENERATOR_API_KEY=...
DATASET_GENERATOR_TIMEOUT_SECONDS=90
DATASET_GENERATOR_MAX_OUTPUT_TOKENS=4000
```

The provider uses Gemini's `models/{model}:generateContent` endpoint with
`contents`, `generationConfig`, and `responseMimeType=application/json`.
Generated cases are stored as drafts and must be approved before they become
trusted benchmark ground truth.

The draft-to-approved workflow is:

```text
Create evaluation dataset
Choose requested case count
Authoring LLM generates draft JSON cases from ingested chunks
Human reviewer checks each question against its supporting chunks
Human reviewer edits question, answer, chunk keys, type, difficulty, or notes
Human reviewer approves or rejects each case
Publish approved cases into an official EvalDataset
Run RAG evaluations only on approved benchmark cases
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
134 passed
```

The tests cover:

- loader validation and deterministic ordering;
- chunk-size limits, overlap, headings, and content preservation;
- embedding input validation, model caching, and cosine similarity;
- repeatable transactional ingestion;
- semantic ranking and edge cases;
- strict prompt construction and citation/refusal rules;
- mock generation and local Ollama integration behavior;
- end-to-end chat pipeline logging;
- custom evaluation metrics, scoring, failure classification, and runner
  persistence;
- single-run scorecards, tuning recommendations, regression comparison
  summaries, and HTML reports;
- internal and HTTP RAG connectors;
- persisted connector configuration and connector tests;
- dataset-authoring LLM provider configuration;
- draft evaluation dataset generation, review, and publishing;
- database foreign-key enforcement;
- API ordering, serialization, chat responses, and error handling;
- idempotent chatbot-version seeding.

External embedding and generation calls are replaced with deterministic test
doubles where needed, keeping the suite fast and independent of network
availability.

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
- **Mock-first generation:** the app can be developed and tested without a local
  LLM, while still supporting Ollama for local generation experiments.
- **Demo-ready container:** the Docker image seeds the corpus, chatbot versions,
  and benchmark dataset during build so deployed demos start with useful data.

## Current limitations

- Retrieval is not yet exposed as a standalone HTTP endpoint.
- The dashboard is a lightweight browser demo, not a full reporting workspace.
- Chatbot-version settings drive `top_k`, generation mode/model, temperature,
  and prompt template. Per-version embedding models and chunk settings are still
  stored for future comparison work, but not applied at query time.
- Chunk embeddings are recomputed for each retrieval request.
- SQLite table creation is used instead of a migration framework.
- The evaluation API queues work with FastAPI background tasks; it is not yet a
  durable worker queue.
- Authentication, pagination, durable background workers, and production
  observability are not implemented yet.

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
- [x] RAG prompt construction
- [x] Answer generator
- [x] End-to-end chat pipeline and API
- [x] Starter evaluation dataset and metrics
- [x] Evaluation runner
- [x] Evaluation API endpoints
- [x] Single-run RAG quality scorecards
- [x] Runtime tuning UI for chatbot versions
- [x] RAG connector interface
- [x] External RAG configuration and test UI
- [x] Separate dataset-authoring LLM configuration
- [x] Draft evaluation dataset generation and human approval workflow
- [x] Editable draft case review with supporting chunks
- [x] Lightweight browser demo
- [x] Stable chunk keys for retrieval-aware evaluation
- [x] Expanded 50-case evaluation dataset
- [x] Regression comparison views
- [x] Dashboard and HTML reports
- [x] Docker deployment packaging
- [ ] CI quality gates

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
