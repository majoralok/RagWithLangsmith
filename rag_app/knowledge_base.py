"""Document loading, chunking, indexing, and similarity retrieval."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import chromadb
from chromadb.errors import NotFoundError
from langsmith import traceable
from pypdf import PdfReader

from rag_app.config import AppSettings

SUPPORTED_EXTENSIONS = {".md", ".pdf", ".txt"}
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@dataclass(frozen=True, slots=True)
class IngestionReport:
    file_count: int
    chunk_count: int


def _trace_retriever_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Exclude the live Chroma object while retaining retrieval configuration."""
    return {
        "query": inputs["query"],
        "top_k": inputs["top_k"],
        "min_relevance_score": inputs["min_relevance_score"],
        "collection_name": inputs["collection_name"],
        "collection_size": inputs["collection_size"],
        "embedding_model": EMBEDDING_MODEL,
        "distance_metric": "cosine",
    }


@traceable(
    name="vector_similarity_search",
    run_type="retriever",
    tags=["rag", "retrieval", "chroma"],
    metadata={"algorithm": "HNSW", "distance_metric": "cosine"},
    process_inputs=_trace_retriever_inputs,
)
def retrieve_documents(
    query: str,
    *,
    collection: Any,
    top_k: int,
    min_relevance_score: float,
    collection_name: str,
    collection_size: int,
) -> list[dict[str, Any]]:
    """Return ranked documents with complete score metadata for LangSmith."""
    result = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection_size),
        include=["documents", "metadatas", "distances"],
    )

    ids = result["ids"][0]
    texts = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    documents: list[dict[str, Any]] = []

    for rank, (chunk_id, text, metadata, distance) in enumerate(
        zip(ids, texts, metadatas, distances), start=1
    ):
        distance_value = float(distance)
        relevance_score = 1.0 - distance_value
        if relevance_score < min_relevance_score:
            continue
        traced_metadata = {
            **(metadata or {}),
            "chunk_id": chunk_id,
            "rank": rank,
            "distance": round(distance_value, 6),
            "relevance_score": round(relevance_score, 6),
        }
        documents.append({"page_content": text, "metadata": traced_metadata})

    return documents


class KnowledgeBase:
    """Owns the persistent Chroma collection and document ingestion lifecycle."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(settings.chroma_path))

    def ingest(self, source: Path, *, replace: bool = True) -> IngestionReport:
        source = source.expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Document source does not exist: {source}")

        files = list(_document_files(source))
        if not files:
            extensions = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(f"No supported documents found. Expected: {extensions}.")

        if replace:
            try:
                self.client.delete_collection(self.settings.collection_name)
            except NotFoundError:
                pass

        collection = self.client.get_or_create_collection(
            name=self.settings.collection_name,
            metadata={
                "embedding_model": EMBEDDING_MODEL,
                "distance_metric": "cosine",
            },
            configuration={"hnsw": {"space": "cosine"}},
        )

        records: list[tuple[str, str, dict[str, Any]]] = []
        root = source if source.is_dir() else source.parent
        for file_path in files:
            source_name = file_path.relative_to(root).as_posix()
            for page_number, text in _read_document(file_path):
                for chunk_index, (chunk, start, end) in enumerate(
                    _chunk_text(
                        text,
                        chunk_size=self.settings.chunk_size,
                        overlap=self.settings.chunk_overlap,
                    ),
                    start=1,
                ):
                    content_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                    chunk_id = hashlib.sha256(
                        f"{source_name}:{page_number}:{chunk_index}:{content_hash}".encode()
                    ).hexdigest()
                    records.append(
                        (
                            chunk_id,
                            chunk,
                            {
                                "source": source_name,
                                "page": page_number,
                                "chunk": chunk_index,
                                "char_start": start,
                                "char_end": end,
                                "content_sha256": content_hash,
                            },
                        )
                    )

        if not records:
            raise ValueError("The documents contained no extractable text.")

        for offset in range(0, len(records), 100):
            batch = records[offset : offset + 100]
            collection.upsert(
                ids=[record[0] for record in batch],
                documents=[record[1] for record in batch],
                metadatas=[record[2] for record in batch],
            )

        return IngestionReport(file_count=len(files), chunk_count=len(records))

    def require_documents(self) -> None:
        collection = self._existing_collection()
        if collection.count() == 0:
            raise RuntimeError("The knowledge base is empty. Run the ingest command first.")

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        collection = self._existing_collection()
        collection_size = collection.count()
        return retrieve_documents(
            query,
            collection=collection,
            top_k=self.settings.top_k,
            min_relevance_score=self.settings.min_relevance_score,
            collection_name=self.settings.collection_name,
            collection_size=collection_size,
        )

    def _existing_collection(self) -> Any:
        try:
            return self.client.get_collection(self.settings.collection_name)
        except NotFoundError as exc:
            raise RuntimeError(
                "No knowledge base exists. Run the ingest command first."
            ) from exc


def _document_files(source: Path) -> Iterator[Path]:
    if source.is_file():
        if source.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield source
        return
    yield from sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _read_document(path: Path) -> Iterator[tuple[int, str]]:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                yield page_number, text
        return

    text = path.read_text(encoding="utf-8", errors="replace")
    if text.strip():
        yield 1, text


def _chunk_text(
    text: str, *, chunk_size: int, overlap: int
) -> Iterator[tuple[str, int, int]]:
    normalized = re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()
    start = 0
    text_length = len(normalized)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            minimum_break = start + chunk_size // 2
            break_candidates = (
                normalized.rfind("\n\n", minimum_break, end),
                normalized.rfind(". ", minimum_break, end),
                normalized.rfind(" ", minimum_break, end),
            )
            natural_break = max(break_candidates)
            if natural_break >= minimum_break:
                end = natural_break + (1 if normalized[natural_break] == "." else 0)

        chunk = normalized[start:end].strip()
        if chunk:
            yield chunk, start, end
        if end >= text_length:
            break
        start = max(end - overlap, start + 1)
