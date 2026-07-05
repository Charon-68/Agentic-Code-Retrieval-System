"""Manual RAG pipeline with fixed-size character splitting and chunk evaluation."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from openai import OpenAI

from src.week1_foundations.config import get_openai_client, get_settings, is_openai_configured
from src.week2_standard_rag.embedder import BaseEmbedder, OpenAIEmbedder
from src.week2_standard_rag.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOC_PATH = PROJECT_ROOT / "data" / "sample_docs" / "sample_wiki.txt"


@dataclass(frozen=True)
class TextChunk:
    """A fragment of source text with positional metadata."""

    text: str
    start_char: int
    end_char: int
    chunk_index: int


class FixedSizeCharacterSplitter:
    """
    Split text into overlapping fixed-size character windows.

  Unlike token-based splitters, character chunking can bisect words and
    sentences, which often degrades retrieval quality — this class supports
    the Week 2 evaluation experiments.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[TextChunk]:
        """Split text into overlapping character chunks."""
        chunks: list[TextChunk] = []
        start = 0
        index = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end]
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                    chunk_index=index,
                )
            )
            if end >= text_len:
                break
            start += self.chunk_size - self.chunk_overlap
            index += 1

        logger.info(
            "Split %d chars into %d chunks (size=%d, overlap=%d)",
            text_len,
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks


@dataclass
class RetrievalResult:
    """A single retrieved chunk with similarity score."""

    text: str
    distance: float
    metadata: dict[str, Any]


class RAGPipeline:
    """End-to-end retrieval-augmented generation without high-level RAG frameworks."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: VectorStore,
        llm_client: Optional[OpenAI] = None,
        llm_model: Optional[str] = None,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm_client = llm_client or get_openai_client()
        self.llm_model = llm_model or get_settings().openai_model

    def ingest(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        source: str = "unknown",
    ) -> int:
        """Chunk, embed, and store a document. Returns number of chunks indexed."""
        splitter = FixedSizeCharacterSplitter(chunk_size, chunk_overlap)
        chunks = splitter.split(text)

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)
        metadatas = [
            {
                "source": source,
                "chunk_index": c.chunk_index,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
            for c in chunks
        ]
        self.vector_store.add_documents(texts, embeddings, metadatas)
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        """Retrieve the most similar chunks for a query."""
        query_embedding = self.embedder.embed_query(query)
        raw = self.vector_store.query(query_embedding, n_results=top_k)

        results: list[RetrievalResult] = []
        documents = raw.get("documents", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]

        for doc, dist, meta in zip(documents, distances, metadatas):
            results.append(
                RetrievalResult(
                    text=doc or "",
                    distance=dist or 0.0,
                    metadata=meta or {},
                )
            )
        return results

    def generate_answer(self, query: str, context_chunks: Sequence[RetrievalResult]) -> str:
        """Synthesize an answer from retrieved context using the LLM."""
        context = "\n\n---\n\n".join(r.text for r in context_chunks)
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the question using only the provided context. "
                        "If the context is insufficient, say so explicitly."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                },
            ],
            temperature=0.0,
            max_tokens=300,
        )
        return response.choices[0].message.content or ""

    def query(self, question: str, top_k: int = 3) -> dict[str, Any]:
        """Full RAG: retrieve then generate."""
        retrieved = self.retrieve(question, top_k=top_k)
        answer = self.generate_answer(question, retrieved)
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": [
                {"text": r.text[:200], "distance": r.distance, "metadata": r.metadata}
                for r in retrieved
            ],
        }


def evaluate_chunk_configurations(
    document_text: str,
    queries: Sequence[str],
    chunk_configs: Sequence[tuple[int, int]],
    source: str = "sample_wiki",
) -> list[dict[str, Any]]:
    """
    Compare retrieval quality across different chunk size / overlap settings.

    Logs how aggressive fragmentation (e.g. 100-char chunks) degrades the
    relevance of retrieved context compared to larger windows (e.g. 1000 chars).
    """
    embedder = OpenAIEmbedder()
    evaluation_results: list[dict[str, Any]] = []

    for chunk_size, chunk_overlap in chunk_configs:
        collection_name = f"eval_{chunk_size}_{chunk_overlap}"
        store = VectorStore(collection_name=collection_name)
        store.reset()

        pipeline = RAGPipeline(embedder=embedder, vector_store=store)
        num_chunks = pipeline.ingest(
            document_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source=source,
        )

        config_result: dict[str, Any] = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "num_chunks": num_chunks,
            "queries": [],
        }

        for query in queries:
            retrieved = pipeline.retrieve(query, top_k=3)
            avg_distance = (
                sum(r.distance for r in retrieved) / len(retrieved) if retrieved else 0.0
            )
            top_chunk_preview = retrieved[0].text[:150] if retrieved else ""

            query_result = {
                "query": query,
                "avg_cosine_distance": round(avg_distance, 4),
                "top_chunk_preview": top_chunk_preview,
                "retrieved_count": len(retrieved),
            }
            config_result["queries"].append(query_result)

            logger.info(
                "chunk_size=%d | query='%s' | avg_distance=%.4f | preview=%s",
                chunk_size,
                query[:50],
                avg_distance,
                top_chunk_preview[:80],
            )

        evaluation_results.append(config_result)

    return evaluation_results


def main(argv: Optional[list[str]] = None) -> int:
    """CLI for Week 2 RAG pipeline and chunk evaluation."""
    parser = argparse.ArgumentParser(description="Week 2 Standard RAG Pipeline")
    parser.add_argument(
        "--mode",
        choices=["query", "evaluate"],
        default="evaluate",
        help="Run a single query or chunk configuration evaluation",
    )
    parser.add_argument(
        "--question",
        default="Who created Python and when was it first released?",
        help="Question for query mode",
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    args = parser.parse_args(argv)

    if not is_openai_configured():
        logger.error("OPENAI_API_KEY not configured. See .env.example")
        return 1

    doc_path = DEFAULT_DOC_PATH
    if not doc_path.exists():
        logger.error("Document not found: %s", doc_path)
        return 1

    document_text = doc_path.read_text(encoding="utf-8")

    if args.mode == "evaluate":
        queries = [
            "Who created Python and when was it first released?",
            "What libraries are used for machine learning in Python?",
            "What is the Global Interpreter Lock?",
            "What is PyPI?",
        ]
        configs = [(100, 20), (500, 50), (1000, 100)]
        results = evaluate_chunk_configurations(document_text, queries, configs)
        print("\n=== Chunk Configuration Evaluation ===\n")
        for r in results:
            print(f"\n--- chunk_size={r['chunk_size']}, overlap={r['chunk_overlap']} ---")
            print(f"Total chunks: {r['num_chunks']}")
            for q in r["queries"]:
                print(f"  Q: {q['query']}")
                print(f"     avg_distance: {q['avg_cosine_distance']}")
                print(f"     preview: {q['top_chunk_preview'][:100]}...")
    else:
        store = VectorStore(collection_name="week2_demo")
        store.reset()
        pipeline = RAGPipeline(embedder=OpenAIEmbedder(), vector_store=store)
        pipeline.ingest(
            document_text,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            source=str(doc_path),
        )
        result = pipeline.query(args.question)
        print(f"\nQuestion: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"\nRetrieved {len(result['retrieved_chunks'])} chunks")

    return 0


if __name__ == "__main__":
    sys.exit(main())
