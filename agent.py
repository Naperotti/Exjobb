"""
agent.py
--------
A simple question-answering agent that uses a local causal language model
to answer natural-language questions.

Usage
-----
    python agent.py                              # interactive REPL mode
    python agent.py --question "What is AI?"     # single-shot mode
    python agent.py --model gpt2 --max-new-tokens 150
    python agent.py --help

The agent prepends a short instruction prefix to each question so that the
causal LM produces answer-style continuations rather than open-ended prose.
"""

import argparse
import sys

import torch

import config
from generate import _load_model


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

_INSTRUCTION_PREFIX = "Answer the following question concisely.\n\nQuestion: {question}\nAnswer:"


def answer_question(
    question: str,
    model_name: str = config.GENERATION_MODEL,
    max_new_tokens: int = config.AGENT_MAX_NEW_TOKENS,
    temperature: float = config.AGENT_TEMPERATURE,
    top_p: float = config.AGENT_TOP_P,
    tokenizer=None,
    model=None,
    device: str | None = None,
) -> str:
    """
    Generate an answer for *question* using a causal LM.

    Parameters
    ----------
    question:
        The natural-language question to answer.
    model_name:
        HuggingFace model identifier (used when *tokenizer*/*model* are None).
    max_new_tokens:
        Maximum number of tokens to generate for the answer.
    temperature:
        Sampling temperature.
    top_p:
        Nucleus-sampling threshold.
    tokenizer / model / device:
        Pre-loaded model objects.  When provided the function skips loading.

    Returns
    -------
    str
        The generated answer text (continuation only, stripped of the prompt).
    """
    if tokenizer is None or model is None:
        tokenizer, model, device = _load_model(model_name)

    prompt = _INSTRUCTION_PREFIX.format(question=question.strip())
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
            num_return_sequences=1,
        )

    answer = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return answer.strip()


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

def run_interactive(
    model_name: str = config.GENERATION_MODEL,
    max_new_tokens: int = config.AGENT_MAX_NEW_TOKENS,
    temperature: float = config.AGENT_TEMPERATURE,
    top_p: float = config.AGENT_TOP_P,
) -> None:
    """Start an interactive question-answering loop."""
    print(f"Loading model: {model_name}")
    tokenizer, model, device = _load_model(model_name)
    print("Agent ready. Type your question and press Enter (Ctrl+C or 'exit' to quit).\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        answer = answer_question(
            question,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        print(f"Agent: {answer}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Question-answering agent powered by a local causal LM."
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question to answer (omit for interactive mode).",
    )
    parser.add_argument(
        "--model",
        default=config.GENERATION_MODEL,
        help="HuggingFace model name (default: %(default)s).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=config.AGENT_MAX_NEW_TOKENS,
        help="Max tokens to generate per answer (default: %(default)s).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=config.AGENT_TEMPERATURE,
        help="Sampling temperature (default: %(default)s).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=config.AGENT_TOP_P,
        help="Top-p nucleus sampling (default: %(default)s).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.question:
        # Single-shot mode
        answer = answer_question(
            args.question,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(answer)
    else:
        # Interactive REPL
        run_interactive(
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )


if __name__ == "__main__":
    main()
