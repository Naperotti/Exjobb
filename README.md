# Exjobb — LLM Embedding Space Analysis

> **Protointerpretation within the embedding space**  
> Locally generate text sequences from a small LLM, encode them into dense
> embeddings, and explore their semantic structure through clustering and
> visualisation.

---

## Overview

This project provides a three-stage pipeline:

```
generate.py  →  embed.py  →  analyze.py
```

| Stage | What it does |
|-------|-------------|
| **Generate** | Prompts a small local causal LM (default: GPT-2) and saves the output sequences to `outputs/sequences.jsonl`. |
| **Embed** | Encodes each sequence into a dense vector using a sentence-transformer (default: `all-MiniLM-L6-v2`). |
| **Analyze** | Reduces to 2-D (UMAP / t-SNE fallback), clusters with k-means, and saves a scatter plot + cluster summary. |

All three stages are wired together in `pipeline.py`.

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# Optional: for UMAP (better than t-SNE for large datasets)
pip install umap-learn
```

### 2. Run the full pipeline

```bash
python pipeline.py
```

Outputs land in `outputs/`:

| File | Description |
|------|-------------|
| `sequences.jsonl` | Generated text sequences (one JSON object per line) |
| `embeddings.npy` | Float32 NumPy array — shape `(N, D)` |
| `metadata.jsonl` | Aligned sequence records (prompt, continuation, model) |
| `embedding_space.png` | 2-D scatter plot coloured by cluster |
| `embedding_space_cluster_summary.json` | Per-cluster stats and example sequences |

### 3. Run stages individually

```bash
# Generate only
python generate.py --model gpt2 --sequences-per-prompt 5 --seed 42

# Embed only (needs sequences.jsonl)
python embed.py --model sentence-transformers/all-MiniLM-L6-v2

# Analyse only (needs embeddings.npy + metadata.jsonl)
python analyze.py --n-clusters 10

# Skip already-completed stages
python pipeline.py --skip-generate --skip-embed
```

### 4. Ask questions (agent)

```bash
# Single-shot: print one answer and exit
python agent.py --question "What is a neural network?"

# Interactive REPL: type questions, get answers, exit with Ctrl+C or 'exit'
python agent.py

# Override model and generation settings
python agent.py --model gpt2 --max-new-tokens 150 --temperature 0.8
```

---

## Configuration

Edit **`config.py`** to change:

- `GENERATION_MODEL` — any HuggingFace causal LM (e.g. `distilgpt2`, `gpt2-medium`)
- `PROMPTS` — seed texts for generation
- `SEQUENCES_PER_PROMPT` — how many continuations per prompt
- `MAX_NEW_TOKENS`, `TEMPERATURE`, `TOP_P` — generation hyper-parameters
- `EMBEDDING_MODEL` — any `sentence-transformers` model
- `N_CLUSTERS` — k-means cluster count
- `PCA_COMPONENTS`, `UMAP_NEIGHBORS`, `UMAP_MIN_DIST` — reduction parameters

---

## Tests

```bash
python -m pytest tests/ -v
```

Tests use lightweight stubs — no model downloads required.

---

## Project structure

```
.
├── config.py          # Central configuration
├── generate.py        # Stage 1 – LLM generation
├── embed.py           # Stage 2 – sentence embedding
├── analyze.py         # Stage 3 – dimensionality reduction, clustering, plot
├── pipeline.py        # End-to-end runner
├── agent.py           # Q&A agent (single-shot & interactive)
├── requirements.txt
├── tests/
│   └── test_pipeline.py
└── outputs/           # Created at runtime
    ├── sequences.jsonl
    ├── embeddings.npy
    ├── metadata.jsonl
    ├── embedding_space.png
    └── embedding_space_cluster_summary.json
```
