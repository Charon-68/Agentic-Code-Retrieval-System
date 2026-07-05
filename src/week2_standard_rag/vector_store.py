"""Persistent ChromaDB vector store for document chunks."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Optional, Sequence

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIR = Path(".chroma")


class VectorStore:
    """Thin wrapper around a persistent ChromaDB collection."""

    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: Path | str = DEFAULT_PERSIST_DIR,
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready at %s",
            collection_name,
            self.persist_directory,
        )

    @property
    def collection(self) -> Collection:
        return self._collection

    def reset(self) -> None:
        """Delete and recreate the collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Reset collection '%s'", self.collection_name)

    def add_documents(
        self,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Optional[Sequence[dict[str, Any]]] = None,
        ids: Optional[Sequence[str]] = None,
    ) -> list[str]:
        """Insert document chunks with precomputed embeddings."""
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have the same length")

        doc_ids = list(ids) if ids else [str(uuid.uuid4()) for _ in texts]
        meta_list = list(metadatas) if metadatas else [{} for _ in texts]

        self._collection.add(
            ids=doc_ids,
            documents=list(texts),
            embeddings=[list(e) for e in embeddings],
            metadatas=meta_list,
        )
        logger.debug("Added %d documents to '%s'", len(texts), self.collection_name)
        return doc_ids

    def query(
        self,
        query_embedding: Sequence[float],
        n_results: int = 5,
        where: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Query the collection by embedding vector."""
        results = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return results

    def count(self) -> int:
        return self._collection.count()
