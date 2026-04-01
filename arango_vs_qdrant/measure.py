"""
Shared measurement utilities and dataclasses.
"""

import time
import statistics
from dataclasses import dataclass, field

import numpy as np

from config import WARMUP_QUERIES


@dataclass
class BenchmarkResult:
    db_name: str
    dataset_size: int
    ingest_time_s: float = 0.0
    index_build_time_s: float = 0.0
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


def measure_durations(query_fn, query_vecs: np.ndarray, k: int) -> list[float]:
    """Run queries sequentially with warm-up, return list of latencies in ms."""
    # Pre-convert all vectors to Python lists so .tolist() cost is excluded from timing
    query_lists = [q.tolist() for q in query_vecs]

    n_warmup = min(WARMUP_QUERIES, len(query_lists))
    for q in query_lists[:n_warmup]:
        query_fn(q, k)

    latencies = []
    for q in query_lists:
        t0 = time.perf_counter()
        query_fn(q, k)
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
        recall = len(ret_set & relevant) / len(relevant)
        recalls.append(recall)
    return statistics.mean(recalls) if recalls else 0.0
