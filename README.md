# Agentic Codebase RAG

A 4-week curriculum repository for building **Retrieval-Augmented Generation (RAG)** systems focused on codebase understanding. Each week introduces progressively advanced techniques — from LLM foundations to hybrid dense/sparse retrieval over GitHub repositories.

## Project Structure

```
.
├── .env.example              # API key placeholders
├── requirements.txt          # Python dependencies
├── data/sample_docs/         # Week 2 document corpus
├── sample_repo/              # Target code for Weeks 3–4
└── src/
    ├── week1_foundations/    # LLM experiments & prompting
    ├── week2_standard_rag/   # Vector RAG from scratch
    ├── week3_ast_rag/        # AST-aware code parsing
    └── week4_hybrid_github/  # GitHub ingestion + hybrid search
```

## Quick Start

### 1. Clone and set up the environment

```bash
cd agentic-codebase-rag
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env with your OPENAI_API_KEY (required) and optional keys
```

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | Yes | Embeddings and chat completions |
| `ANTHROPIC_API_KEY` | No | Week 1 Anthropic client demo |
| `GITHUB_TOKEN` | No | Remote repo ingestion (Week 4) |
| `GITHUB_REPO_OWNER` | No | GitHub org/user for Week 4 |
| `GITHUB_REPO_NAME` | No | Repository name for Week 4 |

### 3. Run from project root

All modules use the `src` package. Set `PYTHONPATH` or run with `-m`:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## Week 1 — Foundations of LLMs

Explores temperature sampling, token logprobs, few-shot classification, chain-of-thought reasoning, and the **Lost in the Middle** phenomenon.

```bash
python -m src.week1_foundations.experiment --task all
python -m src.week1_foundations.experiment --task temperature
python -m src.week1_foundations.experiment --task lost-in-middle
```

**Key files:** `config.py`, `prompts.py`, `experiment.py`

---

## Week 2 — Standard RAG

Builds a manual RAG pipeline with a **fixed-size character splitter**, OpenAI embeddings, and persistent ChromaDB storage. Includes chunk-size evaluation (100 vs 500 vs 1000 characters).

```bash
python -m src.week2_standard_rag.pipeline --mode evaluate
python -m src.week2_standard_rag.pipeline --mode query \
  --question "What is the Global Interpreter Lock?"
```

**Key files:** `embedder.py`, `vector_store.py`, `pipeline.py`

---

## Week 3 — AST-Aware Code RAG

Uses LlamaIndex `CodeHierarchyNodeParser` to preserve class/method boundaries and parent-child scope metadata. Compares AST parsing against Week 2's flat character chunking on `sample_repo/`.

```bash
python -m src.week3_ast_rag.pipeline
python -m src.week3_ast_rag.pipeline \
  --query "RetryPolicy execute method" \
  --flat-chunk-size 150
```

**Key files:** `parser.py`, `pipeline.py`, `sample_repo/`

---

## Week 4 — Hybrid GitHub Search

Ingests a GitHub repository (or falls back to `sample_repo/`) and runs **dense vector + BM25** fusion retrieval. The terminal CLI supports exact keyword lookups and semantic architectural queries.

```bash
# Interactive REPL
python -m src.week4_hybrid_github.app

# Single query
python -m src.week4_hybrid_github.app \
  -q "ERROR_CODE_AUTH_FAILURE" --top-k 3

# With LLM answer synthesis
python -m src.week4_hybrid_github.app \
  -q "How does retry logic work?" --synthesize
```

**Example queries:**
- `ERROR_CODE_AUTH_FAILURE` — exact error code (BM25 excels)
- `RetryPolicy exponential backoff` — class/method keyword match
- `architectural layout of database connection module` — semantic concept

**Key files:** `github_loader.py`, `hybrid_retriever.py`, `app.py`

---

## Design Principles

- **Fully typed** Python with `pydantic` settings validation
- **Async-ready** clients (`AsyncOpenAI`, `AsyncAnthropic`) where applicable
- **dotenv** for secrets — never commit `.env`
- **Structural logging** throughout all pipelines
- **Graceful fallbacks** — local `sample_repo/` when GitHub token is absent

## License

MIT — use freely for learning and experimentation.
