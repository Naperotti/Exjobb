"""
generate.py
-----------
Locally generate text sequences from a small causal language model.

Usage
-----
    python generate.py                   # uses defaults from config.py
    python generate.py --model gpt2     # override the generation model
    python generate.py --help           # show all options

Outputs
-------
Each generated sequence is appended to SEQUENCES_FILE (JSONL format) and
the same metadata is written to METADATA_FILE so that downstream scripts
can track which prompt produced which sequence.
"""

import argparse
import json
import os
import random

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(model_name: str):
    """Load a causal LM and its tokeniser onto the best available device."""
    print(f"Loading generation model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # GPT-2 has no pad token by default; use EOS as pad.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"  → running on {device}")
    return tokenizer, model, device


def generate_sequences(
    prompts: list[str],
    sequences_per_prompt: int = config.SEQUENCES_PER_PROMPT,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    temperature: float = config.TEMPERATURE,
    top_p: float = config.TOP_P,
    model_name: str = config.GENERATION_MODEL,
    seed: int | None = None,
) -> list[dict]:
    """
    Generate *sequences_per_prompt* continuations for each prompt.

    Returns
    -------
    list of dicts with keys:
        - "prompt"   : the seed text
        - "sequence" : full generated text (prompt + continuation)
        - "model"    : model name used for generation
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    tokenizer, model, device = _load_model(model_name)
    results = []

    for prompt in tqdm(prompts, desc="Generating sequences"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=sequences_per_prompt,
            )

        for seq_ids in output_ids:
            full_text = tokenizer.decode(seq_ids, skip_special_tokens=True)
            continuation = tokenizer.decode(
                seq_ids[input_len:], skip_special_tokens=True
            )
            results.append(
                {
                    "prompt": prompt,
                    "continuation": continuation.strip(),
                    "sequence": full_text,
                    "model": model_name,
                }
            )

    return results


def save_sequences(sequences: list[dict], path: str = config.SEQUENCES_FILE) -> None:
    """Persist sequences to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for item in sequences:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(sequences)} sequences → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Generate LLM output sequences.")
    parser.add_argument("--model", default=config.GENERATION_MODEL)
    parser.add_argument("--sequences-per-prompt", type=int, default=config.SEQUENCES_PER_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=config.TOP_P)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=config.SEQUENCES_FILE)
    return parser.parse_args()


def main():
    args = _parse_args()
    sequences = generate_sequences(
        prompts=config.PROMPTS,
        sequences_per_prompt=args.sequences_per_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        model_name=args.model,
        seed=args.seed,
    )
    save_sequences(sequences, args.output)


if __name__ == "__main__":
    main()
