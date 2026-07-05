"""Embedding generation wrapper for OpenAI and optional local models."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Sequence

from openai import AsyncOpenAI, OpenAI

from src.week1_foundations.config import get_async_openai_client, get_openai_client

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS = 1536


class BaseEmbedder(ABC):
    """Abstract interface for text embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts synchronously."""

    @abstractmethod
    async def aembed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts asynchronously."""


class OpenAIEmbedder(BaseEmbedder):
    """Wrapper around OpenAI's text-embedding-3-small (or compatible) API."""

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        client: OpenAI | None = None,
        async_client: AsyncOpenAI | None = None,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self._client = client or get_openai_client()
        self._async_client = async_client or get_async_openai_client()

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        logger.debug("Embedding %d texts with model=%s", len(texts), self.model)
        response = self._client.embeddings.create(
            model=self.model,
            input=list(texts),
            dimensions=self.dimensions,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_texts([query])[0]

    async def aembed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._async_client.embeddings.create(
            model=self.model,
            input=list(texts),
            dimensions=self.dimensions,
        )
        return [item.embedding for item in response.data]

    async def aembed_query(self, query: str) -> list[float]:
        """Asynchronously embed a single query string."""
        embeddings = await self.aembed_texts([query])
        return embeddings[0]


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Local embedding fallback using sentence-transformers.

    Requires: pip install sentence-transformers
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        logger.info("Loaded local SentenceTransformer model: %s", model_name)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(list(texts), convert_to_numpy=True)
        return embeddings.tolist()

    async def aembed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_texts, texts)


def create_embedder(prefer_local: bool = False) -> BaseEmbedder:
    """Factory: OpenAI embedder by default, optional local SentenceTransformer."""
    if prefer_local:
        return SentenceTransformerEmbedder()
    return OpenAIEmbedder()
