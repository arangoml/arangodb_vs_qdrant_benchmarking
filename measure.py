"""
Shared measurement utilities, dataclasses, and synthetic data generation.
"""

import time
import hashlib
import statistics
from dataclasses import dataclass, field

import numpy as np

from config import WARMUP_QUERIES, NUM_CATEGORIES, DIMENSIONS


@dataclass
class BenchmarkResult:
    db_name: str
    dataset_size: int
    index_time_s: float = 0.0
    duration_p50_ms: float = 0.0
    duration_p50_std: float = 0.0
    duration_p95_ms: float = 0.0
    duration_p95_std: float = 0.0
    duration_p99_ms: float = 0.0
    duration_p99_std: float = 0.0
    throughput_qps: float = 0.0
    throughput_qps_std: float = 0.0
    recall_at_k: dict = field(default_factory=dict)      # {k: recall}
    recall_at_k_std: dict = field(default_factory=dict)  # {k: stddev}
    filtered_duration_p50_ms: float = 0.0
    filtered_duration_p50_std: float = 0.0
    filtered_duration_p95_ms: float = 0.0
    filtered_duration_p95_std: float = 0.0
    memory_mb: float = 0.0
    num_runs: int = 1


@dataclass
class RecallDurationPoint:
    """A single point on the recall-vs-duration Pareto curve."""
    db_name: str
    dataset_size: int
    param_name: str   # "nProbe" or "ef"
    param_value: int
    recall_at_10: float
    duration_p50_ms: float
    duration_p95_ms: float


def measure_durations(query_fn, query_vecs: np.ndarray, k: int) -> list[float]:
    """Run queries sequentially with warm-up, return list of latencies in ms."""
    n_warmup = min(WARMUP_QUERIES, len(query_vecs))
    for q in query_vecs[:n_warmup]:
        query_fn(q.tolist(), k)

    latencies = []
    for q in query_vecs:
        t0 = time.perf_counter()
        query_fn(q.tolist(), k)
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def compute_recall_qrels(
    retrieved_ids: list[list[str]],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    k: int,
) -> float:
    """
    Recall@k using human relevance judgments (qrels).
    For each query: what fraction of known-relevant docs appear in top-k?
    """
    recalls = []
    for qid, ret in zip(query_ids, retrieved_ids):
        if qid not in qrels:
            continue
        relevant = {did for did, score in qrels[qid].items() if score > 0}
        if not relevant:
            continue
        ret_set = set(ret[:k])
        recall = len(ret_set & relevant) / min(len(relevant), k)
        recalls.append(recall)
    return statistics.mean(recalls) if recalls else 0.0


def generate_synthetic_batch(
    corpus_embeddings: np.ndarray,
    corpus_texts: list[str],
    rng: np.random.Generator,
    start_index: int,
    count: int,
    dim: int = DIMENSIONS,
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """
    Vectorized batch generation of synthetic docs.
    Returns (embeddings, doc_ids, texts, categories) for `count` docs.
    """
    src_indices = rng.integers(0, len(corpus_embeddings), size=count)

    src_embs = corpus_embeddings[src_indices]  # (count, dim)
    noise = rng.normal(0, 0.01, size=src_embs.shape).astype(np.float32)
    noisy = src_embs + noise
    norms = np.linalg.norm(noisy, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    noisy = noisy / norms

    doc_ids = [f"syn_{start_index + i}" for i in range(count)]
    texts = [corpus_texts[idx] for idx in src_indices]
    categories = [f"cat_{int(hashlib.md5(did.encode()).hexdigest(), 16) % NUM_CATEGORIES}" for did in doc_ids]

    return noisy, doc_ids, texts, categories


def precompute_corpus_arrays(corpus: dict) -> tuple[list[str], np.ndarray, list[str]]:
    """Pre-compute numpy arrays from corpus for fast batch generation."""
    all_ids = sorted(corpus.keys())
    embeddings = np.array(
        [corpus[did]["embedding"] for did in all_ids], dtype=np.float32,
    )
    texts = [f"{corpus[did].get('title', '')}. {corpus[did]['text']}" for did in all_ids]
    return all_ids, embeddings, texts
