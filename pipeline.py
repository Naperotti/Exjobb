"""
pipeline.py
-----------
End-to-end convenience runner: generate → embed → analyze.

Usage
-----
    python pipeline.py                   # full pipeline with config defaults
    python pipeline.py --skip-generate   # re-use existing sequences.jsonl
    python pipeline.py --skip-embed      # re-use existing embeddings.npy
    python pipeline.py --help
"""

import argparse
import os

import config
from generate import generate_sequences, save_sequences
from embed import load_sequences, embed_sequences, save_embeddings
from analyze import analyze


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: generate → embed → analyze."
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip generation and load existing sequences from SEQUENCES_FILE.",
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Skip embedding and load existing embeddings from EMBEDDINGS_FILE.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-clusters", type=int, default=config.N_CLUSTERS)
    args = parser.parse_args()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 – Generate
    # ------------------------------------------------------------------
    if not args.skip_generate:
        print("\n" + "=" * 60)
        print("STEP 1 / 3  –  GENERATE SEQUENCES")
        print("=" * 60)
        sequences = generate_sequences(
            prompts=config.PROMPTS,
            seed=args.seed,
        )
        save_sequences(sequences, config.SEQUENCES_FILE)
    else:
        print(f"\nSkipping generation; using {config.SEQUENCES_FILE}")

    # ------------------------------------------------------------------
    # Step 2 – Embed
    # ------------------------------------------------------------------
    if not args.skip_embed:
        print("\n" + "=" * 60)
        print("STEP 2 / 3  –  EMBED SEQUENCES")
        print("=" * 60)
        records = load_sequences(config.SEQUENCES_FILE)
        embeddings = embed_sequences(records)
        save_embeddings(embeddings, records)
    else:
        print(f"\nSkipping embedding; using {config.EMBEDDINGS_FILE}")

    # ------------------------------------------------------------------
    # Step 3 – Analyse
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3 / 3  –  ANALYSE EMBEDDING SPACE")
    print("=" * 60)
    results = analyze(n_clusters=args.n_clusters)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  Sequences   : {config.SEQUENCES_FILE}")
    print(f"  Embeddings  : {config.EMBEDDINGS_FILE}")
    print(f"  Plot        : {config.PLOT_FILE}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
