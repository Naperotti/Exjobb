"""
embed.py
--------
Encode generated text sequences into dense embedding vectors using a
sentence-transformer model.

Usage
-----
    python embed.py                          # use defaults from config.py
    python embed.py --input outputs/sequences.jsonl
    python embed.py --model sentence-transformers/all-MiniLM-L6-v2

Outputs
-------
- EMBEDDINGS_FILE  : float32 NumPy array of shape (N, D), one row per sequence.
- METADATA_FILE    : JSONL mirror of the sequence records (prompt, sequence, …).
"""

import argparse
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sequences(path: str = config.SEQUENCES_FILE) -> list[dict]:
    """Load sequences from a JSONL file."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} sequences from {path}")
    return records


def embed_sequences(
    records: list[dict],
    model_name: str = config.EMBEDDING_MODEL,
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode the 'sequence' field of each record into a dense vector.

    Returns
    -------
    embeddings : np.ndarray, shape (N, D)
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [r["sequence"] for r in records]
    print(f"Encoding {len(texts)} sequences …")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # unit-norm → cosine similarity ≡ dot product
    )
    return embeddings.astype(np.float32)


def save_embeddings(
    embeddings: np.ndarray,
    records: list[dict],
    embeddings_path: str = config.EMBEDDINGS_FILE,
    metadata_path: str = config.METADATA_FILE,
) -> None:
    """Save embeddings (.npy) and aligned metadata (.jsonl)."""
    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings {embeddings.shape} → {embeddings_path}")

    with open(metadata_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved metadata → {metadata_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Embed generated sequences.")
    parser.add_argument("--input", default=config.SEQUENCES_FILE)
    parser.add_argument("--model", default=config.EMBEDDING_MODEL)
    parser.add_argument("--embeddings-out", default=config.EMBEDDINGS_FILE)
    parser.add_argument("--metadata-out", default=config.METADATA_FILE)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main():
    args = _parse_args()
    records = load_sequences(args.input)
    embeddings = embed_sequences(records, model_name=args.model, batch_size=args.batch_size)
    save_embeddings(embeddings, records, args.embeddings_out, args.metadata_out)


if __name__ == "__main__":
    main()
