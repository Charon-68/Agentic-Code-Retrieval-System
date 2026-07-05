"""Hybrid dense vector + sparse BM25 retrieval using LlamaIndex."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import Document, NodeWithScore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.retrievers.bm25 import BM25Retriever

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid retrieval fusion."""

    vector_top_k: int = 5
    bm25_top_k: int = 5
    fusion_top_k: int = 5
    fusion_mode: str = "reciprocal_rerank"
    similarity_threshold: float = 0.0


def configure_llama_index_settings(
    embedding_model: str = "text-embedding-3-small",
    llm_model: str = "gpt-4o-mini",
) -> None:
    """Configure global LlamaIndex embedding and LLM models."""
    Settings.embed_model = OpenAIEmbedding(model=embedding_model)
    Settings.llm = LlamaOpenAI(model=llm_model, temperature=0.0)


class HybridRetriever:
    """
    Combines dense vector search with sparse BM25 lexical matching.

    Fusion retriever merges ranked results so exact keyword matches (error codes,
    class/method names) surface alongside semantically similar chunks.
    """

    def __init__(
        self,
        documents: list[Document],
        config: Optional[HybridSearchConfig] = None,
    ) -> None:
        self.config = config or HybridSearchConfig()
        self.documents = documents

        configure_llama_index_settings()
        self._index = VectorStoreIndex.from_documents(documents, show_progress=True)
        self._vector_retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=self.config.vector_top_k,
        )
        self._bm25_retriever = BM25Retriever.from_defaults(
            index=self._index,
            similarity_top_k=self.config.bm25_top_k,
        )
        self._fusion_retriever = QueryFusionRetriever(
            retrievers=[self._vector_retriever, self._bm25_retriever],
            similarity_top_k=self.config.fusion_top_k,
            num_queries=1,
            mode=self.config.fusion_mode,
            use_async=False,
        )
        self._query_engine = RetrieverQueryEngine(
            retriever=self._fusion_retriever,
            response_synthesizer=get_response_synthesizer(),
        )
        logger.info(
            "HybridRetriever ready | vector_k=%d bm25_k=%d fusion_k=%d",
            self.config.vector_top_k,
            self.config.bm25_top_k,
            self.config.fusion_top_k,
        )

    @property
    def index(self) -> VectorStoreIndex:
        return self._index

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[NodeWithScore]:
        """Run hybrid fusion retrieval without LLM synthesis."""
        k = top_k or self.config.fusion_top_k
        self._fusion_retriever.similarity_top_k = k
        return self._fusion_retriever.retrieve(query)

    def query(self, query: str) -> str:
        """Run full hybrid RAG query with LLM answer synthesis."""
        response = self._query_engine.query(query)
        return str(response)

    def retrieve_with_references(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve fused results with file/line references for terminal display.

        Line numbers are extracted from node metadata when available.
        """
        nodes = self.retrieve(query, top_k=top_k)
        results: list[dict[str, Any]] = []

        for rank, node_with_score in enumerate(nodes, start=1):
            node = node_with_score.node
            meta = node.metadata or {}
            file_path = (
                meta.get("file_path")
                or meta.get("file_name")
                or meta.get("filename")
                or "unknown"
            )
            start_line = meta.get("start_line") or meta.get("line_number")
            end_line = meta.get("end_line")
            line_ref = (
                f"{file_path}:{start_line}-{end_line}"
                if start_line and end_line
                else f"{file_path}:{start_line}"
                if start_line
                else file_path
            )

            results.append(
                {
                    "rank": rank,
                    "score": round(node_with_score.score or 0.0, 4),
                    "file_path": file_path,
                    "line_reference": line_ref,
                    "scope_path": meta.get("scope_path"),
                    "text": node.get_content(),
                }
            )

        return results
