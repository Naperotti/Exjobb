"""
tests/test_pipeline.py
----------------------
Unit tests for the generation → embedding → analysis pipeline.

These tests use lightweight stubs / small tensors so they run quickly
without downloading any actual model weights.
"""

import json
import os
import sys
import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Minimal stubs so the tests don't need GPU or downloaded models
# ---------------------------------------------------------------------------

# -- Stub for transformers --------------------------------------------------
transformers_stub = types.ModuleType("transformers")


class _FakeBatchEncoding(dict):
    """dict subclass that also exposes a .to() method (like BatchEncoding)."""

    def to(self, device):
        return self


class _FakeTokenizer:
    pad_token = None
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, text, return_tensors="pt"):
        import torch
        return _FakeBatchEncoding(
            {
                "input_ids": torch.zeros(1, 5, dtype=torch.long),
                "attention_mask": torch.ones(1, 5, dtype=torch.long),
            }
        )

    def decode(self, ids, skip_special_tokens=True):
        return "the quick brown fox"

    @classmethod
    def from_pretrained(cls, name):
        return cls()


class _FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        import torch
        n = kwargs.get("num_return_sequences", 1)
        return torch.zeros(n, 10, dtype=torch.long)

    @classmethod
    def from_pretrained(cls, name):
        return cls()


transformers_stub.AutoTokenizer = _FakeTokenizer
transformers_stub.AutoModelForCausalLM = _FakeModel
sys.modules.setdefault("transformers", transformers_stub)

# -- Stub for sentence_transformers -----------------------------------------
st_stub = types.ModuleType("sentence_transformers")


class _FakeSentenceTransformer:
    def __init__(self, name):
        pass

    def encode(self, texts, **kwargs):
        rng = np.random.default_rng(0)
        return rng.random((len(texts), 384), dtype=np.float32)


st_stub.SentenceTransformer = _FakeSentenceTransformer
sys.modules.setdefault("sentence_transformers", st_stub)

# ---------------------------------------------------------------------------
# Now import project modules
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config  # noqa: E402
from generate import generate_sequences, save_sequences  # noqa: E402
from embed import load_sequences, embed_sequences, save_embeddings  # noqa: E402
from analyze import (  # noqa: E402
    reduce_pca,
    cluster_kmeans,
    build_cluster_summary,
    plot_embedding_space,
)


# ---------------------------------------------------------------------------
# Tests – generate
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_returns_correct_count(self):
        prompts = ["Hello world", "Test prompt"]
        seqs_per = 3
        results = generate_sequences(prompts, sequences_per_prompt=seqs_per)
        assert len(results) == len(prompts) * seqs_per

    def test_record_has_required_keys(self):
        results = generate_sequences(["A prompt"], sequences_per_prompt=1)
        record = results[0]
        for key in ("prompt", "continuation", "sequence", "model"):
            assert key in record, f"Missing key: {key}"

    def test_save_and_reload(self, tmp_path):
        path = str(tmp_path / "seqs.jsonl")
        records = [{"prompt": "p", "continuation": "c", "sequence": "p c", "model": "gpt2"}]
        save_sequences(records, path)
        loaded = load_sequences(path)
        assert loaded == records


# ---------------------------------------------------------------------------
# Tests – embed
# ---------------------------------------------------------------------------

class TestEmbed:
    def _make_records(self, n=8):
        return [
            {"prompt": f"prompt {i}", "continuation": "cont",
             "sequence": f"seq {i}", "model": "m"}
            for i in range(n)
        ]

    def test_embedding_shape(self):
        records = self._make_records(8)
        embs = embed_sequences(records, show_progress=False)
        assert embs.shape == (8, 384)
        assert embs.dtype == np.float32

    def test_save_and_reload_embeddings(self, tmp_path):
        records = self._make_records(4)
        embs = embed_sequences(records, show_progress=False)
        emb_path = str(tmp_path / "embs.npy")
        meta_path = str(tmp_path / "meta.jsonl")
        save_embeddings(embs, records, emb_path, meta_path)

        loaded = np.load(emb_path)
        assert loaded.shape == embs.shape
        assert np.allclose(loaded, embs)

        with open(meta_path) as fh:
            lines = [json.loads(ln) for ln in fh if ln.strip()]
        assert len(lines) == len(records)


# ---------------------------------------------------------------------------
# Tests – analyze
# ---------------------------------------------------------------------------

class TestAnalyze:
    def _random_embeddings(self, n=30, d=384, seed=0):
        rng = np.random.default_rng(seed)
        return rng.random((n, d), dtype=np.float32)

    def _make_metadata(self, n=30):
        prompts = [f"prompt {i % 5}" for i in range(n)]
        return [
            {"prompt": p, "sequence": f"{p} continuation {i}", "model": "gpt2"}
            for i, p in enumerate(prompts)
        ]

    def test_pca_reduces_dimensions(self):
        embs = self._random_embeddings(30, 384)
        reduced = reduce_pca(embs, 10)
        assert reduced.shape == (30, 10)

    def test_pca_clamps_to_min_dim(self):
        embs = self._random_embeddings(5, 20)
        reduced = reduce_pca(embs, 100)   # ask for more than available
        assert reduced.shape[0] == 5
        assert reduced.shape[1] <= 20

    def test_cluster_kmeans_labels(self):
        embs = self._random_embeddings(30, 384)
        labels = cluster_kmeans(embs, 5)
        assert labels.shape == (30,)
        assert set(labels).issubset(set(range(5)))

    def test_cluster_summary_structure(self):
        embs = self._random_embeddings(20, 384)
        metadata = self._make_metadata(20)
        labels = cluster_kmeans(embs, 4)
        summary = build_cluster_summary(labels, metadata, embs)
        assert len(summary) == len(set(labels))
        for entry in summary:
            assert "cluster" in entry
            assert "size" in entry
            assert "top_prompts" in entry
            assert entry["size"] > 0

    def test_plot_saves_file(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        embs = self._random_embeddings(20, 384)
        metadata = self._make_metadata(20)
        labels = cluster_kmeans(embs, 4)
        coords = np.random.default_rng(0).random((20, 2))
        plot_path = str(tmp_path / "plot.png")
        plot_embedding_space(coords, labels, metadata, plot_path, n_clusters=4)
        assert os.path.exists(plot_path)
        assert os.path.getsize(plot_path) > 0
