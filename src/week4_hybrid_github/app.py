"""Terminal interface for hybrid codebase search."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

from src.week1_foundations.config import is_openai_configured
from src.week4_hybrid_github.github_loader import (
    DEFAULT_LOCAL_FALLBACK,
    GitHubRepoConfig,
    load_repository_documents,
)
from src.week4_hybrid_github.hybrid_retriever import HybridRetriever, HybridSearchConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║         Agentic Codebase RAG — Hybrid Search CLI             ║
║   Semantic concepts  +  Exact keywords (BM25 fusion)         ║
╚══════════════════════════════════════════════════════════════╝
"""

EXAMPLE_QUERIES = [
    "ERROR_CODE_AUTH_FAILURE",
    "RetryPolicy exponential backoff",
    "How does DatabaseConnection handle retries?",
    "validate_config missing keys",
    "architectural layout of database connection module",
]


def format_results(results: list[dict], max_preview: int = 400) -> str:
    """Format retrieval results for terminal output."""
    if not results:
        return "  (no results)"

    lines: list[str] = []
    for item in results:
        preview = item["text"][:max_preview]
        if len(item["text"]) > max_preview:
            preview += "..."

        lines.append(f"\n  [{item['rank']}] score={item['score']} | {item['line_reference']}")
        if item.get("scope_path"):
            lines.append(f"      scope: {item['scope_path']}")
        lines.append("      ─────────────────────────────────────")
        for line in preview.splitlines():
            lines.append(f"      {line}")

    return "\n".join(lines)


class CodebaseSearchApp:
    """Interactive terminal runtime for hybrid codebase queries."""

    def __init__(
        self,
        retriever: HybridRetriever,
        source: str,
    ) -> None:
        self.retriever = retriever
        self.source = source

    def run_query(self, query: str, top_k: int = 5, synthesize: bool = False) -> None:
        """Execute a single query and print fused results."""
        print(f"\n🔍 Query: {query}")
        print(f"   Source: {self.source}")
        print("─" * 60)

        results = self.retriever.retrieve_with_references(query, top_k=top_k)
        print(format_results(results))

        if synthesize and is_openai_configured():
            print("\n💡 Synthesized Answer:")
            print("─" * 60)
            answer = self.retriever.query(query)
            print(f"   {answer}")

    def run_interactive(self, top_k: int = 5, synthesize: bool = False) -> None:
        """REPL loop for continuous querying."""
        print(BANNER)
        print(f"Indexed source: {self.source}")
        print("\nExample queries:")
        for i, q in enumerate(EXAMPLE_QUERIES, 1):
            print(f"  {i}. {q}")
        print("\nCommands: 'quit'/'exit' to leave, 'examples' to reprint samples\n")

        while True:
            try:
                user_input = input("codebase> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "q"}:
                print("Goodbye.")
                break
            if user_input.lower() == "examples":
                for i, q in enumerate(EXAMPLE_QUERIES, 1):
                    print(f"  {i}. {q}")
                continue

            self.run_query(user_input, top_k=top_k, synthesize=synthesize)


def build_app(
    github_config: Optional[GitHubRepoConfig] = None,
    local_fallback: Path = DEFAULT_LOCAL_FALLBACK,
    search_config: Optional[HybridSearchConfig] = None,
) -> CodebaseSearchApp:
    """Load documents and construct the search application."""
    documents, source = load_repository_documents(
        config=github_config,
        local_fallback=local_fallback,
    )
    if not documents:
        raise RuntimeError("No documents loaded for indexing")

    retriever = HybridRetriever(documents=documents, config=search_config)
    return CodebaseSearchApp(retriever=retriever, source=source)


async def arun_query(app: CodebaseSearchApp, query: str, top_k: int = 5) -> None:
    """Async wrapper for non-blocking query execution in async contexts."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, app.run_query, query, top_k, False)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for Week 4 hybrid search app."""
    parser = argparse.ArgumentParser(
        description="Hybrid codebase search — vector + BM25 fusion",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query (omit for interactive mode)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of fused results to return",
    )
    parser.add_argument(
        "--synthesize",
        action="store_true",
        help="Also generate an LLM-synthesized answer",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=DEFAULT_LOCAL_FALLBACK,
        help="Local directory fallback when GitHub is unavailable",
    )
    args = parser.parse_args(argv)

    if not is_openai_configured():
        logger.error("OPENAI_API_KEY not configured. See .env.example")
        return 1

    try:
        app = build_app(local_fallback=args.local_dir)
    except Exception as exc:
        logger.error("Failed to initialize app: %s", exc)
        return 1

    if args.query:
        app.run_query(args.query, top_k=args.top_k, synthesize=args.synthesize)
    else:
        app.run_interactive(top_k=args.top_k, synthesize=args.synthesize)

    return 0


if __name__ == "__main__":
    sys.exit(main())
