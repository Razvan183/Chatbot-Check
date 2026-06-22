# ChatbotCheck

ChatbotCheck is a quality-assurance platform for retrieval-augmented generation
(RAG) chatbots. It will test whether chatbot answers are grounded in retrieved
documents, properly cited, safe to release, and improving between versions.

The project is intentionally built in small, understandable stages. The current
stage provides document loading, paragraph-aware chunking, SQLite persistence,
repeatable ingestion, and semantic retrieval over the sample policy corpus.

For a complete architecture walkthrough, module-by-module reference, current
roadmap status, and technical-interview preparation, see
[`PROJECT_TECHNICAL_GUIDE.md`](PROJECT_TECHNICAL_GUIDE.md).

## Current features

- FastAPI application entry point
- `GET /health` endpoint
- Interactive API documentation
- Environment-based configuration
- SQLAlchemy models backed by SQLite
- Seven fictional company-policy documents
- UTF-8 Markdown and text loading
- Bounded, overlapping document chunking
- Sentence-transformer embedding utilities
- Cosine-similarity retrieval over stored chunks
- Four repeatable demo chatbot configurations
- Repeatable document ingestion
- Read-only document and chunk API endpoints
- Automated tests for the implemented behavior

## Prerequisites

- Python 3.10 or newer

You can check your Python version with:

```powershell
python --version
```

On Windows, if `python` is unavailable but the Python launcher is installed,
use `py` in place of `python` in the commands below.

## Local setup

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start the API:

```powershell
python -m uvicorn app.main:app --reload
```

Then open:

- Health check: <http://127.0.0.1:8000/health>
- Interactive API documentation: <http://127.0.0.1:8000/docs>

The health endpoint should return:

```json
{
  "status": "ok",
  "service": "evalforge-api"
}
```

## Configuration

Application settings live in `app/config.py`. Each setting has a local default,
so a `.env` file is optional.

To customize the settings, copy the example file:

```powershell
Copy-Item .env.example .env
```

You can then edit `.env` without changing the Python source. The real `.env`
file is ignored by Git because it may eventually contain local secrets, while
`.env.example` documents the variables the application supports.

## Database

ChatbotCheck currently uses SQLite through SQLAlchemy. When the API starts, it
creates `evalforge.db` and any missing tables automatically.

The database file is local generated state, so it is excluded from Git. Table
definitions live in `app/db/models.py`; connection and session setup live in
`app/db/database.py`.

## Inspecting source documents

The ingestion loader reads UTF-8 Markdown and text files into Python
dictionaries containing the filename, source path, document type, and text.

Run the inspection script from the project root:

```powershell
python -m scripts.inspect_documents
```

## Running tests

Run the test suite from the project root:

```powershell
python -m pytest
```

The chunker keeps normal paragraphs together, enforces a configurable character
limit, and adds as much trailing context as safely fits in the next chunk.

The embedding utilities load the configured sentence-transformer model lazily
and cache it for reuse. The first real embedding request may download the model
if it is not already available locally.

The retriever embeds a question together with the stored document chunks,
ranks the chunks by cosine similarity, and returns the most relevant results
with their document metadata.

## Ingesting documents

Load the policy files, split them into chunks, and save them to SQLite:

```powershell
python -m scripts.ingest_documents
```

The script replaces previously ingested demo documents, so it is safe to rerun
after editing a policy or changing the chunk settings.

Direct execution also works:

```powershell
python scripts/ingest_documents.py
```

## Creating chatbot versions

Create or refresh the four demo RAG configurations:

```powershell
python -m scripts.create_demo_versions
```

The script is safe to rerun. Existing demo versions are updated in place so
their configuration remains consistent without creating duplicate records.

## Document API

After ingesting documents and starting the API, these endpoints are available:

```text
GET /documents
GET /documents/{document_id}
GET /documents/{document_id}/chunks
```

Use <http://127.0.0.1:8000/docs> to explore and call them interactively.
