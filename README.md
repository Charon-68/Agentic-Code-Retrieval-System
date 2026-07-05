# Agentic Code Retrieval System

This repository contains a compact educational project for building an Agentic Retrieval-Augmented Generation (RAG) system over software repositories. The implementation currently covers Weeks 1–4 and focuses on prompt engineering, retrieval pipelines, AST-aware code chunking, and hybrid repo search.

## Project Context

The project was developed as part of the Summer of Science (SoS) 2026 initiative for Generative and Agentic AI.

- Project: CS20 – Generative and Agentic AI
- Student: Shourya Saxena
- Mentor: Anuj Kushwaha
- Duration: May 2026 – July 2026

## What is implemented today

### Week 1 – LLM Foundations

- OpenAI and Anthropic API integration
- Temperature sampling experiments
- Few-shot prompting
- Chain-of-thought prompting
- Lost-in-the-middle experiment
- Token probability inspection

### Week 2 – Standard RAG

- Fixed-size document chunking
- OpenAI embeddings
- Persistent ChromaDB vector store
- Cosine similarity retrieval
- Chunk-size evaluation

### Week 3 – AST-aware Code Retrieval

- LlamaIndex CodeHierarchyNodeParser
- Function-level and class-level semantic chunking
- Parent-child scope preservation
- AST-aware retrieval comparison

### Week 4 – Hybrid Repository Search

- GitHub repository ingestion
- Local repository fallback
- Hybrid vector + BM25 retrieval
- Interactive CLI-based exploration
- Semantic and keyword search
- Optional LLM-based answer synthesis

### Future work

- Weeks 5–8 are not implemented in this repository and remain future work.

## Repository layout

```text
.
├── assets/
├── configs/
├── data/
│   └── sample_docs/
├── docs/
│   └── architecture.md
├── sample_repo/
├── scripts/
├── src/
│   ├── week1_foundations/
│   ├── week2_standard_rag/
│   ├── week3_ast_rag/
│   └── week4_hybrid_github/
├── tests/
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
├── requirements.txt
```

## Installation

```bash
git clone https://github.com/<username>/Agentic-Code-Retrieval-System.git
cd Agentic-Code-Retrieval-System
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Required environment variables:

```text
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GITHUB_TOKEN=
GITHUB_REPO_OWNER=
GITHUB_REPO_NAME=
```

## Usage

### Week 1

```bash
python -m src.week1_foundations.experiment --task all
```

### Week 2

```bash
python -m src.week2_standard_rag.pipeline --mode evaluate
```

### Week 3

```bash
python -m src.week3_ast_rag.pipeline
```

### Week 4

```bash
python -m src.week4_hybrid_github.app
```

Or with a single query:

```bash
python -m src.week4_hybrid_github.app -q "Where is retry logic implemented?"
```

## Example questions

- Explain the RetryPolicy implementation.
- Where is authentication handled?
- Which files import connection.py?
- Describe the repository architecture.
- What is the purpose of utils.py?
- Find exponential backoff implementation.

## Technology stack

- LLM providers: OpenAI, Anthropic
- Frameworks: LlamaIndex
- Retrieval: ChromaDB, BM25, hybrid retrieval

## Parsing

- CodeHierarchyNodeParser
- Tree-sitter

## Configuration

- Pydantic
- python-dotenv

---

# Future Work

The remaining phases of the project focus on transforming the retrieval pipeline into a fully autonomous software engineering assistant through:

- Event-driven agent workflows
- Tool orchestration
- Multi-step planning
- Code impact analysis
- Refactoring recommendation
- Repository-wide dependency reasoning

---

# License

This project is released under the MIT License.
