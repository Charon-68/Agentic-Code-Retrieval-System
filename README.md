# Agentic Code Retrieval System

> **An Agentic Retrieval-Augmented Generation (RAG) framework for semantic understanding of software repositories using Hybrid Search, AST-aware code parsing, and autonomous reasoning workflows.**

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-Framework-green)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

# Overview

Modern software repositories contain thousands of files, making navigation difficult using traditional search techniques.

Conventional **keyword search** struggles to understand semantic intent, while **pure vector search** often misses exact identifiers such as function names, class names, constants, and module imports.

This project explores how modern **Large Language Models (LLMs)** can understand software repositories by progressively building an intelligent retrieval system capable of:

- Understanding natural language developer queries
- Retrieving relevant code using semantic and lexical search
- Preserving program structure through AST-aware parsing
- Answering repository-level questions using Retrieval-Augmented Generation (RAG)
- Evolving into a fully autonomous software engineering agent through ReAct-style reasoning workflows

The project is being developed as part of the **Summer of Science (SoS) 2026** program.

**Project:** CS20 – Generative and Agentic AI

**Student:** Shourya Saxena

**Mentor:** Anuj Kushwaha

---

# Features

## Implemented

### LLM Foundations

- OpenAI GPT integration
- Anthropic Claude integration
- Temperature sampling experiments
- Few-shot prompting
- Chain-of-Thought prompting
- Lost-in-the-Middle evaluation
- Token probability inspection

---

### Retrieval-Augmented Generation

- Fixed-size document chunking
- OpenAI embeddings
- Persistent ChromaDB vector database
- Cosine similarity retrieval
- Chunk size evaluation
- Retrieval benchmarking

---

### AST-aware Code Understanding

- LlamaIndex CodeHierarchyNodeParser
- Function-level semantic chunking
- Class-level semantic chunking
- Parent-child scope preservation
- AST vs flat chunk comparison

---

### Hybrid Repository Search

- GitHub repository ingestion
- Local repository fallback
- Hybrid Vector + BM25 retrieval
- Reciprocal Rank Fusion
- Interactive CLI
- Semantic repository search
- Keyword search
- LLM answer synthesis

---

### Engineering

- Modular Python package architecture
- Typed configuration using Pydantic
- Environment-based configuration
- Unit test framework
- Documentation

---

# Current Progress

| Module | Status |
|---------|:------:|
| Week 1 – LLM Foundations | ✅ |
| Week 2 – Standard RAG | ✅ |
| Week 3 – AST-aware Retrieval | ✅ |
| Week 4 – Hybrid GitHub Search | ✅ |
| Week 5 – Agent Workflow | 🚧 In Progress |
| Week 6 – Tool Integration | ⏳ Planned |
| Week 7 – Code Analysis Agent | ⏳ Planned |
| Week 8 – Optimization & Evaluation | ⏳ Planned |

---

# System Architecture

```
                     User Query
                          │
                          ▼
                 Query Processing
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
 Semantic Retrieval                Keyword Retrieval
 (Dense Embeddings)                     (BM25)
          │                               │
          └───────────────┬───────────────┘
                          ▼
                 Hybrid Fusion Retriever
                          │
                          ▼
                 Relevant Code Chunks
                          │
                          ▼
                Large Language Model
                          │
                          ▼
                  Generated Response
```

The upcoming agentic workflow extends this architecture by introducing intelligent routing and multi-step reasoning.

```
                     User Query
                          │
                          ▼
                  Agent Workflow
                          │
                 Intent Classification
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
 Hybrid Search      AST Retrieval      File Reader
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
                  Observation Memory
                          │
                 Additional Tool Needed?
                    │               │
                   Yes             No
                    │               ▼
                    └──────► Final Response
```

---

# Repository Structure

```
.
├── assets/                     # Images and diagrams
├── configs/                    # Future configuration files
├── data/
│   └── sample_docs/            # Sample documents for RAG
├── docs/
│   └── architecture.md         # Detailed architecture
├── sample_repo/                # Sample repository indexed by the system
├── scripts/                    # Utility scripts
├── src/
│   ├── week1_foundations/
│   ├── week2_standard_rag/
│   ├── week3_ast_rag/
│   ├── week4_hybrid_github/
│   └── week5_agent/            # Agent workflow (in progress)
├── tests/
├── README.md
├── requirements.txt
└── pyproject.toml
```

---

# Project Evolution

## Week 1 — Foundations of LLMs

Studied the behavior of Large Language Models through prompt engineering experiments.

Implemented:

- Temperature sampling
- Few-shot prompting
- Chain-of-Thought
- Lost-in-the-Middle experiment

---

## Week 2 — Standard Retrieval-Augmented Generation

Built a complete RAG pipeline over unstructured documents using ChromaDB.

Implemented:

- Document chunking
- Embeddings
- Vector indexing
- Semantic retrieval
- Chunk evaluation

---

## Week 3 — AST-aware Code Retrieval

Extended RAG to software repositories using Abstract Syntax Trees.

Implemented:

- CodeHierarchyNodeParser
- Semantic code chunking
- Scope preservation
- AST vs Flat retrieval comparison

---

## Week 4 — Hybrid Repository Search

Combined semantic retrieval with lexical search over GitHub repositories.

Implemented:

- GitHub ingestion
- Hybrid retrieval
- Reciprocal Rank Fusion
- Interactive CLI
- Repository question answering

---

## Week 5 — Agent Workflow (In Progress)

Currently implementing:

- LlamaIndex Workflows
- Agent state management
- Intent routing
- Event-driven execution
- Tool orchestration
- ReAct reasoning loop

---

# Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/Agentic-Code-Retrieval-System.git

cd Agentic-Code-Retrieval-System
```

Create a virtual environment

```bash
python -m venv .venv

source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create your environment file

```bash
cp .env.example .env
```

Required environment variables

```text
OPENAI_API_KEY=

ANTHROPIC_API_KEY=

GITHUB_TOKEN=

GITHUB_REPO_OWNER=

GITHUB_REPO_NAME=
```

---

# Usage

## Week 1

```bash
python -m src.week1_foundations.experiment --task all
```

---

## Week 2

```bash
python -m src.week2_standard_rag.pipeline --mode evaluate
```

Query mode

```bash
python -m src.week2_standard_rag.pipeline \
    --mode query \
    --question "What is the Global Interpreter Lock?"
```

---

## Week 3

```bash
python -m src.week3_ast_rag.pipeline
```

---

## Week 4

Interactive CLI

```bash
python -m src.week4_hybrid_github.app
```

Single query

```bash
python -m src.week4_hybrid_github.app \
-q "Where is retry logic implemented?"
```

With answer synthesis

```bash
python -m src.week4_hybrid_github.app \
-q "Explain RetryPolicy." \
--synthesize
```

---

# Example Questions

- Where is RetryPolicy implemented?
- Explain the authentication flow.
- Describe the database connection architecture.
- Which files import `connection.py`?
- Where is exponential backoff implemented?
- Explain the repository structure.
- Which module handles retries?
- What is the purpose of `utils.py`?

---

# Technology Stack

| Category | Technology |
|-----------|------------|
| Programming Language | Python 3.11 |
| LLM Providers | OpenAI GPT, Anthropic Claude |
| Agent Framework | LlamaIndex |
| Vector Database | ChromaDB |
| Retrieval | BM25, Reciprocal Rank Fusion |
| Embeddings | OpenAI Embeddings |
| Code Parsing | CodeHierarchyNodeParser |
| Parsing Backend | Tree-sitter |
| Configuration | Pydantic |
| Environment Management | python-dotenv |

---

# Testing

Run the complete test suite

```bash
pytest
```

Run a specific module

```bash
pytest tests/test_week2.py
```

---

# Roadmap

### Week 5

- Event-driven workflows
- Agent state management
- Intent routing
- Tool orchestration
- ReAct reasoning

### Week 6

- Hybrid retrieval tool integration
- AST retrieval integration
- Multi-step repository question answering

### Week 7

- Code analysis agent
- Refactoring recommendations
- Dependency reasoning
- Cascading failure prediction

### Week 8

- Performance optimization
- Large repository benchmarking
- Evaluation metrics
- Documentation
- Final synthesis

---

# References

- LlamaIndex
- OpenAI API
- Anthropic API
- ChromaDB
- Tree-sitter
- BM25
- Reciprocal Rank Fusion (RRF)

---

# License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for details.
