"""Command-line entry point for the RAG application."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load tracing variables before importing modules decorated by LangSmith.
load_dotenv()

from rag_app.config import AppSettings, ConfigurationError  # noqa: E402
from rag_app.knowledge_base import KnowledgeBase  # noqa: E402
from rag_app.pipeline import answer_question  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local RAG with an open-source Hugging Face LLM and LangSmith traces."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser(
        "ingest", help="Index .pdf, .md, and .txt files."
    )
    ingest.add_argument("source", type=Path, help="A document file or directory.")
    ingest.add_argument(
        "--append",
        action="store_true",
        help="Keep the existing collection and upsert these documents.",
    )

    ask = subparsers.add_parser("ask", help="Ask one question.")
    ask.add_argument("question", help="Question to answer from the indexed documents.")
    ask.add_argument(
        "--show-context", action="store_true", help="Print retrieved chunks and scores."
    )

    chat = subparsers.add_parser("chat", help="Start an interactive question loop.")
    chat.add_argument(
        "--show-context", action="store_true", help="Print retrieved chunks and scores."
    )
    return parser


def print_result(result: dict, show_context: bool) -> None:
    print(f"\n{result['answer']}\n")
    if result["sources"]:
        print("Sources:")
        for source in result["sources"]:
            page = f", page {source['page']}" if source.get("page") else ""
            print(
                f"  [{source['rank']}] {source['source']}{page} "
                f"(score: {source['relevance_score']:.4f})"
            )

    if show_context and result["documents"]:
        print("\nRetrieved context:")
        for document in result["documents"]:
            metadata = document["metadata"]
            print(
                f"\n--- rank={metadata['rank']} "
                f"score={metadata['relevance_score']:.4f} "
                f"distance={metadata['distance']:.4f} ---"
            )
            print(document["page_content"])


def ask_once(
    question: str, settings: AppSettings, knowledge_base: KnowledgeBase, show_context: bool
) -> None:
    result = answer_question(question, settings, knowledge_base)
    print_result(result, show_context)


def main() -> int:
    args = build_parser().parse_args()
    settings = AppSettings.from_environment()
    knowledge_base = KnowledgeBase(settings)

    if args.command == "ingest":
        report = knowledge_base.ingest(args.source, replace=not args.append)
        print(
            f"Indexed {report.chunk_count} chunks from {report.file_count} files "
            f"into '{settings.collection_name}'."
        )
        return 0

    settings.require_hugging_face_token()
    knowledge_base.require_documents()

    if args.command == "ask":
        ask_once(args.question, settings, knowledge_base, args.show_context)
        return 0

    tracing = "enabled" if settings.langsmith_tracing_enabled else "disabled"
    print(f"Interactive RAG (LangSmith tracing: {tracing}). Type 'exit' to stop.")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            ask_once(question, settings, knowledge_base, args.show_context)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ConfigurationError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

