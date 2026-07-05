# System Architecture

## Overview

The Agentic Code Retrieval System is organized as a modular pipeline where each project phase builds on the previous one. The current implementation covers the retrieval workflow for Weeks 1–4, while Weeks 5–8 are intentionally left as future work.

## High-level architecture

```text
User Query
    │
    ▼
Query Processing Layer
    │
    ├── Semantic Retrieval (vector embeddings)
    └── Keyword Retrieval (BM25)
          │
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

## Repository components

### Week 1

Purpose: study the behavior of modern LLMs.

Components:

- Prompt engineering
- Temperature sampling
- Few-shot prompting
- Chain-of-thought prompting
- Lost-in-the-middle experiment

### Week 2

Purpose: build a standard retrieval-augmented generation pipeline.

Pipeline:

- Document chunking
- Embedding generation
- ChromaDB vector storage
- Similarity search

### Week 3

Purpose: improve retrieval quality for source code.

Approach:

- AST-based parsing
- Function-level and class-level chunking
- Hierarchical code structure preservation

### Week 4

Purpose: improve retrieval quality through hybrid search.

Approach:

- GitHub repository ingestion
- Local repository fallback
- Dense retrieval + BM25 fusion
- Interactive querying

## Data flow

```text
Repository / Documents
    │
    ▼
Reader
    │
    ▼
Parser
    │
    ▼
Chunks
    │
    ▼
Embeddings
    │
    ▼
Vector Store
    │
    ▼
Retriever
    │
    ▼
Context
    │
    ▼
LLM
    │
    ▼
Answer
```

## Current directory responsibilities

```text
src/
├── week1_foundations/
├── week2_standard_rag/
├── week3_ast_rag/
└── week4_hybrid_github/
```

## Future direction

The repository is intentionally scoped to the currently implemented retrieval and search workflows. Any agentic workflow, tool routing, or multi-step reasoning layers would be introduced in future iterations rather than as placeholder modules.

The following components are planned for Weeks 5–8.

```
                    User Query
                         │
                         ▼
                 Workflow Engine
          (LlamaIndex Workflows)
                         │
                Intent Classification
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
 Hybrid Search      AST Search      File Reader
        │                │                │
        └────────────────┼────────────────┘
                         ▼
                 Observation Memory
                         │
                 Additional Tool?
                    │          │
                   Yes        No
                    │          ▼
                    └────► Response Generator
```

---

# Planned Tool Interfaces

Hybrid Retriever

```
retrieve(query)
```

AST Search

```
search_ast(query)
```

File Reader

```
read_file(path)
```

Dependency Search

```
find_dependencies(symbol)
```

Code Analyzer

```
analyze_module(path)
```

Impact Analyzer

```
predict_changes(function)
```

---

# Design Principles

The system follows the following engineering principles:

- Modular architecture
- Independent retrieval components
- Separation of indexing and reasoning
- Reusable tool interfaces
- Extensible workflow design
- Event-driven orchestration
- Scalable repository indexing

These principles allow future agentic workflows to reuse the retrieval infrastructure without redesigning the underlying system.
