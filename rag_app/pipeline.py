"""Retrieval-augmented generation pipeline with nested LangSmith spans."""

from __future__ import annotations

from typing import Any

from huggingface_hub import InferenceClient
from langsmith import traceable

from rag_app.config import AppSettings
from rag_app.knowledge_base import KnowledgeBase

SYSTEM_PROMPT = """You answer questions using only the supplied context.
If the context does not contain the answer, say that you do not have enough information.
Cite supporting passages with [Source N]. Do not invent facts or citations."""


def _llm_trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": inputs["messages"],
        "model": inputs["model"],
        "provider": inputs["provider"],
        "temperature": inputs["temperature"],
        "max_tokens": inputs["max_tokens"],
    }


@traceable(
    name="hugging_face_chat_completion",
    run_type="llm",
    tags=["rag", "generation", "hugging-face"],
    process_inputs=_llm_trace_inputs,
)
def generate_answer(
    *,
    client: InferenceClient,
    messages: list[dict[str, str]],
    model: str,
    provider: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    response = client.chat_completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    choice = response.choices[0]
    usage = getattr(response, "usage", None)
    usage_details = (
        {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
        if usage is not None
        else None
    )
    return {
        "answer": choice.message.content or "",
        "model": getattr(response, "model", model),
        "finish_reason": choice.finish_reason,
        "usage": usage_details,
    }


def _pipeline_trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    settings: AppSettings = inputs["settings"]
    return {
        "question": inputs["question"],
        "model": settings.hf_model,
        "provider": settings.hf_provider,
        "collection": settings.collection_name,
    }


@traceable(
    name="rag_pipeline",
    run_type="chain",
    tags=["rag"],
    process_inputs=_pipeline_trace_inputs,
)
def answer_question(
    question: str, settings: AppSettings, knowledge_base: KnowledgeBase
) -> dict[str, Any]:
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    documents = knowledge_base.retrieve(question)
    context = _format_context(documents)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]

    client = InferenceClient(
        provider=settings.hf_provider,
        api_key=settings.hf_token,
        timeout=60,
    )
    generation = generate_answer(
        client=client,
        messages=messages,
        model=settings.hf_model,
        provider=settings.hf_provider,
        temperature=settings.llm_temperature,
        max_tokens=settings.max_new_tokens,
    )
    sources = [
        {
            "rank": document["metadata"]["rank"],
            "source": document["metadata"]["source"],
            "page": document["metadata"].get("page"),
            "chunk": document["metadata"].get("chunk"),
            "relevance_score": document["metadata"]["relevance_score"],
        }
        for document in documents
    ]
    return {
        "answer": generation["answer"],
        "sources": sources,
        "documents": documents,
        "generation": {
            "model": generation["model"],
            "finish_reason": generation["finish_reason"],
            "usage": generation["usage"],
        },
    }


def _format_context(documents: list[dict[str, Any]]) -> str:
    if not documents:
        return "No relevant context was retrieved."

    sections = []
    for document in documents:
        metadata = document["metadata"]
        page = f", page {metadata['page']}" if metadata.get("page") else ""
        sections.append(
            f"[Source {metadata['rank']}: {metadata['source']}{page}]\n"
            f"{document['page_content']}"
        )
    return "\n\n".join(sections)

