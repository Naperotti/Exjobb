"""Central configuration for generation and embedding parameters."""

# ---------------------------------------------------------------------------
# Generation model
# ---------------------------------------------------------------------------
# Any causal-LM available on HuggingFace Hub works here.
# "gpt2" (~124 M parameters) is a good default: small, fast, freely available.
GENERATION_MODEL = "gpt2"

# Seed prompts used to kick off generation.
# Extend this list to explore different semantic regions.
PROMPTS = [
    "The purpose of language is",
    "A neural network learns",
    "In the beginning there was",
    "Science helps us understand",
    "Human consciousness emerges from",
    "The relationship between words and meaning",
    "Mathematics is the language of",
    "Memory and experience shape",
    "The future of artificial intelligence",
    "Emotions influence decision making because",
]

# How many independent sequences to generate per prompt.
SEQUENCES_PER_PROMPT = 5

# Maximum number of *new* tokens to generate (beyond the prompt).
MAX_NEW_TOKENS = 60

# Sampling temperature (higher → more creative / diverse).
TEMPERATURE = 0.9

# Top-p (nucleus) sampling threshold.
TOP_P = 0.95

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
# "all-MiniLM-L6-v2" is a lightweight (~22 M) sentence-transformer that
# produces 384-dimensional embeddings and runs well on CPU.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Analysis / visualisation
# ---------------------------------------------------------------------------
# Number of PCA components used before UMAP (set to None to skip PCA pre-step).
PCA_COMPONENTS = 50

# Number of UMAP neighbours (controls local vs. global structure).
UMAP_NEIGHBORS = 15

# Minimum distance between UMAP points.
UMAP_MIN_DIST = 0.1

# Target 2-D embedding for visualisation.
UMAP_N_COMPONENTS = 2

# Number of k-means clusters.
N_CLUSTERS = len(PROMPTS)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = "outputs"
SEQUENCES_FILE = f"{OUTPUT_DIR}/sequences.jsonl"
EMBEDDINGS_FILE = f"{OUTPUT_DIR}/embeddings.npy"
METADATA_FILE = f"{OUTPUT_DIR}/metadata.jsonl"
PLOT_FILE = f"{OUTPUT_DIR}/embedding_space.png"
