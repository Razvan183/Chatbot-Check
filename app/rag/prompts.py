"""Prompt construction for retrieval-augmented policy answers."""

from typing import Any


REFUSAL_MESSAGE = "I could not find this information in the provided documents."
DEFAULT_PROMPT_TEMPLATE = """You are a company policy assistant.

Answer the user's question using only the provided context.

If the answer is not present in the context, say:
"{refusal_message}"

Do not invent policies, numbers, benefits, dates, or procedures.

Include citations using the format [chunk_id].

Context:
{context}

Question:
{question}

Answer:"""


def _validate_retrieved_chunk(chunk: dict[str, Any]) -> None:
    """Ensure a retrieved chunk has the fields needed by the prompt."""
    if not isinstance(chunk, dict):
        raise ValueError("retrieved_chunks must contain dictionaries")

    required_fields = ("chunk_id", "filename", "chunk_text")
    if any(field not in chunk for field in required_fields):
        raise ValueError("retrieved_chunks must include chunk_id, filename, and chunk_text")

    if not str(chunk["chunk_text"]).strip():
        raise ValueError("retrieved_chunks must include non-empty chunk_text values")


def _format_context(retrieved_chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks as citation-ready context blocks."""
    if not retrieved_chunks:
        return "No relevant context was retrieved."

    context_blocks: list[str] = []
    for chunk in retrieved_chunks:
        _validate_retrieved_chunk(chunk)
        chunk_id = chunk["chunk_id"]
        filename = chunk["filename"]
        chunk_text = str(chunk["chunk_text"]).strip()
        context_blocks.append(f"[{chunk_id}] {filename}\n{chunk_text}")

    return "\n\n".join(context_blocks)


def build_rag_prompt(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    prompt_template: str | None = None,
) -> str:
    """Build a strict RAG prompt from a user question and retrieved chunks."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(retrieved_chunks, list):
        raise ValueError("retrieved_chunks must be a list")
    if prompt_template is not None and (
        not isinstance(prompt_template, str) or not prompt_template.strip()
    ):
        raise ValueError("prompt_template must be a non-empty string")

    context = _format_context(retrieved_chunks)
    template = prompt_template or DEFAULT_PROMPT_TEMPLATE

    try:
        return template.format(
            context=context,
            question=question.strip(),
            refusal_message=REFUSAL_MESSAGE,
        )
    except KeyError as exc:
        raise ValueError(f"Unknown prompt template placeholder: {exc.args[0]}") from exc
