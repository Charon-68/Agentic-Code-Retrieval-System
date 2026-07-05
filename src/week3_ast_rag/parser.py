"""AST-aware code parsing using LlamaIndex CodeHierarchyNodeParser."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Optional

from llama_index.core import Document
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.utils import get_tokenizer


def _load_code_hierarchy_parser_class():
    """Load the parser class without triggering the deprecated package initializer."""
    module_name = "llama_index.packs.code_hierarchy.code_hierarchy"
    try:
        module = __import__(module_name, fromlist=["CodeHierarchyNodeParser"])
    except Exception:
        module_path = None
        for base in sys.path:
            candidate = Path(base) / "llama_index" / "packs" / "code_hierarchy" / "code_hierarchy.py"
            if candidate.exists():
                module_path = candidate
                break
        if module_path is None:
            raise
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load parser module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.CodeHierarchyNodeParser


CodeHierarchyNodeParser = _load_code_hierarchy_parser_class()

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CODE_DIR = PROJECT_ROOT / "sample_repo"

SUPPORTED_LANGUAGES = {"python", "javascript", "typescript", "java", "go"}


def load_code_documents(code_dir: Path) -> list[Document]:
    """Load Python source files from a directory as LlamaIndex Documents."""
    documents: list[Document] = []
    if not code_dir.exists():
        raise FileNotFoundError(f"Code directory not found: {code_dir}")

    for file_path in sorted(code_dir.rglob("*.py")):
        if file_path.name.startswith("."):
            continue
        text = file_path.read_text(encoding="utf-8")
        rel_path = file_path.relative_to(code_dir)
        documents.append(
            Document(
                text=text,
                metadata={
                    "file_path": str(rel_path),
                    "absolute_path": str(file_path),
                    "language": "python",
                },
            )
        )
        logger.debug("Loaded document: %s (%d chars)", rel_path, len(text))

    logger.info("Loaded %d Python files from %s", len(documents), code_dir)
    return documents


def create_code_hierarchy_parser(
    language: str = "python",
    max_chars: int = 1500,
) -> CodeHierarchyNodeParser:
    """
    Initialize CodeHierarchyNodeParser for the target language.

    The parser preserves class/function boundaries and parent-child
    relationships in the AST hierarchy metadata.
    """
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{language}'. Choose from: {SUPPORTED_LANGUAGES}"
        )

    parser = CodeHierarchyNodeParser.from_defaults(
        language=language,
        max_chars=max_chars,
        tokenizer=get_tokenizer(),
    )
    logger.info(
        "Created CodeHierarchyNodeParser language=%s max_chars=%d",
        language,
        max_chars,
    )
    return parser


def parse_code_to_nodes(
    code_dir: Optional[Path] = None,
    language: str = "python",
    max_chars: int = 1500,
) -> list[BaseNode]:
    """
    Parse all Python files in code_dir into AST-aware semantic nodes.

    Each node carries hierarchy metadata linking children to parent scopes
    (e.g., methods inside classes).
    """
    target_dir = code_dir or DEFAULT_CODE_DIR
    documents = load_code_documents(target_dir)
    parser = create_code_hierarchy_parser(language=language, max_chars=max_chars)
    nodes = parser.get_nodes_from_documents(documents)
    logger.info("Parsed %d AST-aware nodes from %s", len(nodes), target_dir)
    return nodes


def summarize_node_hierarchy(nodes: list[BaseNode]) -> list[dict[str, Any]]:
    """Extract hierarchy metadata from parsed nodes for inspection."""
    summaries: list[dict[str, Any]] = []

    for node in nodes:
        meta = node.metadata or {}
        summaries.append(
            {
                "node_id": node.node_id,
                "file_path": meta.get("file_path", "unknown"),
                "hierarchy_level": meta.get("hierarchy_level"),
                "parent_hierarchy": meta.get("parent_hierarchy"),
                "scope_path": meta.get("scope_path"),
                "char_count": len(node.get_content(metadata_mode=MetadataMode.NONE)),
                "preview": node.get_content(metadata_mode=MetadataMode.NONE)[:120],
            }
        )

    return summaries
