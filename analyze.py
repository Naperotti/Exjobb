"""
analyze.py
----------
Analyse the semantic embedding space of LLM-generated sequences.

Steps
-----
1. Load embeddings + metadata produced by embed.py.
2. (Optional) Reduce dimensionality with PCA for numerical stability.
3. Project to 2-D with UMAP (falls back to t-SNE if umap-learn is not installed).
4. Cluster with k-means.
5. Save an annotated scatter plot and a cluster-summary JSON.

Usage
-----
    python analyze.py
    python analyze.py --n-clusters 5 --plot outputs/myplot.png
"""

import argparse
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

import config


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def reduce_pca(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    """Reduce embeddings to *n_components* dimensions using PCA."""
    n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA: {embeddings.shape[1]}D → {n_components}D "
          f"(explained variance: {explained:.1%})")
    return reduced


def reduce_2d(embeddings: np.ndarray, n_neighbors: int, min_dist: float) -> np.ndarray:
    """
    Project embeddings to 2-D.

    Tries UMAP first; falls back to t-SNE if umap-learn is not available.
    """
    try:
        import umap  # type: ignore
        print("Projecting to 2-D with UMAP …")
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric="cosine",
            random_state=42,
        )
        return reducer.fit_transform(embeddings)
    except ImportError:
        warnings.warn(
            "umap-learn is not installed; falling back to t-SNE. "
            "Install with: pip install umap-learn",
            stacklevel=2,
        )
        from sklearn.manifold import TSNE
        print("Projecting to 2-D with t-SNE …")
        perplexity = min(30, max(5, embeddings.shape[0] // 5))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        return tsne.fit_transform(embeddings)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_kmeans(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """Fit k-means and return cluster labels."""
    n_clusters = min(n_clusters, embeddings.shape[0])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = km.fit_predict(embeddings)
    if embeddings.shape[0] > n_clusters:
        score = silhouette_score(embeddings, labels, sample_size=min(500, len(labels)))
        print(f"K-means (k={n_clusters}) silhouette score: {score:.3f}")
    return labels


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_embedding_space(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    metadata: list[dict],
    plot_path: str = config.PLOT_FILE,
    n_clusters: int = config.N_CLUSTERS,
) -> None:
    """Save a scatter plot of the 2-D embedding space coloured by cluster."""
    fig, ax = plt.subplots(figsize=(12, 8))

    cmap = plt.get_cmap("tab20", n_clusters)
    scatter = ax.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=labels,
        cmap=cmap,
        alpha=0.75,
        s=60,
        linewidths=0.3,
        edgecolors="white",
    )

    # Annotate a random sample of points with their source prompt.
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(metadata), size=min(20, len(metadata)), replace=False)
    for idx in sample_idx:
        prompt_short = metadata[idx]["prompt"][:30] + "…"
        ax.annotate(
            prompt_short,
            xy=coords_2d[idx],
            fontsize=6,
            alpha=0.7,
            xytext=(4, 4),
            textcoords="offset points",
        )

    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title("LLM Output Embedding Space", fontsize=14)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved → {plot_path}")


# ---------------------------------------------------------------------------
# Cluster summary
# ---------------------------------------------------------------------------

def build_cluster_summary(
    labels: np.ndarray,
    metadata: list[dict],
    embeddings: np.ndarray,
) -> list[dict]:
    """Build a per-cluster summary: size, dominant prompts, centroid distance stats."""
    summaries = []
    unique_labels = sorted(set(labels))
    for lbl in unique_labels:
        mask = labels == lbl
        members = [metadata[i] for i in range(len(metadata)) if mask[i]]
        prompt_counts: dict[str, int] = {}
        for m in members:
            prompt_counts[m["prompt"]] = prompt_counts.get(m["prompt"], 0) + 1
        top_prompts = sorted(prompt_counts.items(), key=lambda x: -x[1])[:3]

        # Intra-cluster cosine similarity via unit-normed embeddings.
        cluster_embs = normalize(embeddings[mask])
        sim_matrix = cluster_embs @ cluster_embs.T
        if sim_matrix.shape[0] > 1:
            upper = sim_matrix[np.triu_indices(sim_matrix.shape[0], k=1)]
            avg_sim = float(upper.mean())
        else:
            avg_sim = 1.0

        summaries.append(
            {
                "cluster": int(lbl),
                "size": int(mask.sum()),
                "top_prompts": [{"prompt": p, "count": c} for p, c in top_prompts],
                "avg_intra_cluster_cosine_sim": avg_sim,
                "example_sequences": [m["sequence"][:120] for m in members[:3]],
            }
        )
    return summaries


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def analyze(
    embeddings_path: str = config.EMBEDDINGS_FILE,
    metadata_path: str = config.METADATA_FILE,
    plot_path: str = config.PLOT_FILE,
    n_clusters: int = config.N_CLUSTERS,
    pca_components: int | None = config.PCA_COMPONENTS,
    umap_neighbors: int = config.UMAP_NEIGHBORS,
    umap_min_dist: float = config.UMAP_MIN_DIST,
) -> dict:
    """
    Full analysis pipeline.

    Returns a dict with keys: 'coords_2d', 'labels', 'summary'.
    """
    # Load data
    embeddings = np.load(embeddings_path)
    metadata: list[dict] = []
    with open(metadata_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                metadata.append(json.loads(line))
    print(f"Loaded embeddings {embeddings.shape} and {len(metadata)} metadata records.")

    # PCA pre-step
    reduced = embeddings
    if pca_components is not None and embeddings.shape[1] > pca_components:
        reduced = reduce_pca(embeddings, pca_components)

    # 2-D projection
    coords_2d = reduce_2d(reduced, umap_neighbors, umap_min_dist)

    # Clustering (on original high-dim embeddings for better cluster quality)
    labels = cluster_kmeans(embeddings, n_clusters)

    # Visualise
    plot_embedding_space(coords_2d, labels, metadata, plot_path, n_clusters)

    # Cluster summary
    summary = build_cluster_summary(labels, metadata, embeddings)
    summary_path = plot_path.replace(".png", "_cluster_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"Cluster summary → {summary_path}")

    return {"coords_2d": coords_2d, "labels": labels, "summary": summary}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Analyse LLM embedding space.")
    parser.add_argument("--embeddings", default=config.EMBEDDINGS_FILE)
    parser.add_argument("--metadata", default=config.METADATA_FILE)
    parser.add_argument("--plot", default=config.PLOT_FILE)
    parser.add_argument("--n-clusters", type=int, default=config.N_CLUSTERS)
    parser.add_argument("--pca-components", type=int, default=config.PCA_COMPONENTS)
    parser.add_argument("--umap-neighbors", type=int, default=config.UMAP_NEIGHBORS)
    parser.add_argument("--umap-min-dist", type=float, default=config.UMAP_MIN_DIST)
    return parser.parse_args()


def main():
    args = _parse_args()
    analyze(
        embeddings_path=args.embeddings,
        metadata_path=args.metadata,
        plot_path=args.plot,
        n_clusters=args.n_clusters,
        pca_components=args.pca_components,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
    )


if __name__ == "__main__":
    main()
