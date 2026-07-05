"""AST-aware RAG pipeline with comparison against flat character chunking."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.embeddings.openai import OpenAIEmbedding

from src.week1_foundations.config import is_openai_configured
from src.week2_standard_rag.pipeline import FixedSizeCharacterSplitter
from src.week3_ast_rag.parser import (
    DEFAULT_CODE_DIR,
    load_code_documents,
    parse_code_to_nodes,
    summarize_node_hierarchy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _configure_llama_index() -> None:
    """Set global LlamaIndex embedding model from OpenAI."""
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        embed_batch_size=32,
    )


def build_ast_index(
    code_dir: Optional[Path] = None,
    max_chars: int = 1500,
) -> tuple[VectorStoreIndex, list[BaseNode]]:
    """Build a vector index from AST-parsed code nodes."""
    nodes = parse_code_to_nodes(code_dir=code_dir, max_chars=max_chars)
    index = VectorStoreIndex(nodes, show_progress=True)
    return index, nodes


def build_flat_chunk_index(
    code_dir: Optional[Path] = None,
    chunk_size: int = 200,
    chunk_overlap: int = 20,
) -> VectorStoreIndex:
    """
    Build a baseline index using Week 2's fixed-size character splitter.

    Small chunk sizes intentionally fragment class/method boundaries to
    demonstrate retrieval degradation vs. AST-aware parsing.
    """
    target_dir = code_dir or DEFAULT_CODE_DIR
    documents = load_code_documents(target_dir)
    splitter = FixedSizeCharacterSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    nodes: list[BaseNode] = []
    from llama_index.core.schema import TextNode

    for doc in documents:
        chunks = splitter.split(doc.text)
        for chunk in chunks:
            nodes.append(
                TextNode(
                    text=chunk.text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": chunk.chunk_index,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "parser": "flat_character",
                    },
                )
            )

    logger.info("Built %d flat character chunks from %s", len(nodes), target_dir)
    return VectorStoreIndex(nodes, show_progress=True)


def query_index(
    index: VectorStoreIndex,
    query: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Query a vector index and return ranked results with metadata."""
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)

    results: list[dict[str, Any]] = []
    for node_with_score in nodes:
        node = node_with_score.node
        results.append(
            {
                "score": round(node_with_score.score or 0.0, 4),
                "file_path": (node.metadata or {}).get("file_path", "unknown"),
                "scope_path": (node.metadata or {}).get("scope_path"),
                "hierarchy_level": (node.metadata or {}).get("hierarchy_level"),
                "parser": (node.metadata or {}).get("parser", "ast_hierarchy"),
                "text_preview": node.get_content(metadata_mode=MetadataMode.NONE)[:300],
            }
        )
    return results


def compare_ast_vs_flat(
    queries: list[str],
    code_dir: Optional[Path] = None,
    flat_chunk_size: int = 200,
) -> dict[str, Any]:
    """
    Run identical queries against AST and flat indexes.

    Demonstrates that AST parsing preserves semantic code blocks (classes,
    methods) while flat chunking often splits them across fragments.
    """
    _configure_llama_index()
    target = code_dir or DEFAULT_CODE_DIR

    ast_index, ast_nodes = build_ast_index(code_dir=target)
    flat_index = build_flat_chunk_index(
        code_dir=target,
        chunk_size=flat_chunk_size,
        chunk_overlap=20,
    )

    comparison: dict[str, Any] = {
        "code_dir": str(target),
        "ast_node_count": len(ast_nodes),
        "flat_chunk_size": flat_chunk_size,
        "hierarchy_summary": summarize_node_hierarchy(ast_nodes),
        "query_results": [],
    }

    for query in queries:
        ast_results = query_index(ast_index, query)
        flat_results = query_index(flat_index, query)

        ast_has_scope = any(r.get("scope_path") for r in ast_results)
        flat_fragmented = any(
            "class " in (r["text_preview"] or "") and "def " not in (r["text_preview"] or "")
            for r in flat_results
        )

        entry = {
            "query": query,
            "ast_top_result": ast_results[0] if ast_results else None,
            "flat_top_result": flat_results[0] if flat_results else None,
            "ast_preserves_scope_metadata": ast_has_scope,
            "flat_likely_fragmented": flat_fragmented,
            "ast_all_results": ast_results,
            "flat_all_results": flat_results,
        }
        comparison["query_results"].append(entry)

        logger.info("Query: %s", query)
        if ast_results:
            logger.info(
                "  AST top | file=%s scope=%s | %s",
                ast_results[0]["file_path"],
                ast_results[0].get("scope_path"),
                ast_results[0]["text_preview"][:80],
            )
        if flat_results:
            logger.info(
                "  Flat top | file=%s | %s",
                flat_results[0]["file_path"],
                flat_results[0]["text_preview"][:80],
            )

    return comparison


def main(argv: Optional[list[str]] = None) -> int:
    """CLI for Week 3 AST RAG comparison."""
    parser = argparse.ArgumentParser(description="Week 3 AST-Aware Code RAG")
    parser.add_argument(
        "--code-dir",
        type=Path,
        default=DEFAULT_CODE_DIR,
        help="Directory containing Python source to index",
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Query to run (can be repeated)",
    )
    parser.add_argument(
        "--flat-chunk-size",
        type=int,
        default=200,
        help="Character chunk size for flat baseline comparison",
    )
    args = parser.parse_args(argv)

    if not is_openai_configured():
        logger.error("OPENAI_API_KEY not configured. See .env.example")
        return 1

    default_queries = [
        "RetryPolicy execute method with exponential backoff",
        "DatabaseConnection connect method and session handling",
        "ERROR_CODE_AUTH_FAILURE authentication error",
        "validate_config required keys validation",
    ]
    queries = args.queries or default_queries

    results = compare_ast_vs_flat(
        queries=queries,
        code_dir=args.code_dir,
        flat_chunk_size=args.flat_chunk_size,
    )

    print("\n=== Week 3: AST vs Flat Chunking Comparison ===\n")
    print(f"Code directory: {results['code_dir']}")
    print(f"AST nodes: {results['ast_node_count']}")
    print(f"Flat chunk size: {results['flat_chunk_size']}")

    print("\n--- AST Hierarchy Summary ---")
    for item in results["hierarchy_summary"][:10]:
        print(
            f"  {item['file_path']} | level={item['hierarchy_level']} "
            f"| scope={item.get('scope_path')} | {item['preview'][:60]}..."
        )

    print("\n--- Query Comparison ---")
    for entry in results["query_results"]:
        print(f"\nQ: {entry['query']}")
        ast_top = entry.get("ast_top_result") or {}
        flat_top = entry.get("flat_top_result") or {}
        print(f"  AST  | score={ast_top.get('score')} scope={ast_top.get('scope_path')}")
        print(f"       | {ast_top.get('text_preview', '')[:120]}")
        print(f"  Flat | score={flat_top.get('score')}")
        print(f"       | {flat_top.get('text_preview', '')[:120]}")
        print(f"  AST preserves scope metadata: {entry['ast_preserves_scope_metadata']}")
        print(f"  Flat likely fragmented: {entry['flat_likely_fragmented']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
