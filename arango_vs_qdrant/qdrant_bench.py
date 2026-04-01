"""
Qdrant setup, querying, ingestion, and benchmark runner.
"""

import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np

from config import (
    QDRANT_HOST, QDRANT_HTTP_PORT, QDRANT_GRPC_PORT,
    DIMENSIONS, TOP_K_VALUES, CONTAINER_MEMORY_GB,
    BATCH_SIZE, CONCURRENT_WORKERS, NUM_CATEGORIES, NUM_RUNS,
)
from docker import get_container_memory_usage_mb
from measure import (
    BenchmarkResult,
    measure_durations, compute_recall_qrels,
)


def make_qdrant_client():
    """Create a new Qdrant gRPC client (one per thread for thread safety)."""
    from qdrant_client import QdrantClient
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_HTTP_PORT, grpc_port=QDRANT_GRPC_PORT, prefer_grpc=True, timeout=1200)


def wait_for_qdrant(timeout: int = 120):
    """Block until Qdrant is reachable."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://{QDRANT_HOST}:{QDRANT_HTTP_PORT}/healthz", timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"Qdrant not reachable at {QDRANT_HOST}:{QDRANT_HTTP_PORT} after {timeout}s")


def setup_qdrant(dim: int, m: int | None = None, ef_construct: int | None = None):
    from qdrant_client.models import Distance, VectorParams, HnswConfigDiff

    wait_for_qdrant()
    client = make_qdrant_client()
    collection = "documents"

    if client.collection_exists(collection):
        client.delete_collection(collection)

    kwargs = {}
    if m is not None or ef_construct is not None:
        kwargs["hnsw_config"] = HnswConfigDiff(
            m=m or 16, ef_construct=ef_construct or 100,
        )

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        **kwargs,
    )
    return client, collection


def query_qdrant(client, collection: str, query_vec: list[float], k: int,
                 category_filter: str | None = None,
                 ef: int | None = None) -> list[str]:
    from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchParams

    search_filter = None
    if category_filter:
        search_filter = Filter(
            must=[FieldCondition(key="category", match=MatchValue(value=category_filter))]
        )

    search_params = None
    if ef is not None:
        search_params = SearchParams(hnsw_ef=ef)

    results = client.query_points(
        collection_name=collection,
        query=query_vec,
        limit=k,
        query_filter=search_filter,
        search_params=search_params,
    )
    return [hit.payload["doc_id"] for hit in results.points]


def measure_throughput_qdrant(collection: str, query_vecs: np.ndarray, k: int,
                              category_filter: str | None = None,
                              ef: int | None = None,
                              workers: int = CONCURRENT_WORKERS) -> float:
    """Run Qdrant queries concurrently with per-thread clients."""
    import threading

    thread_local = threading.local()

    def _get_client():
        if not hasattr(thread_local, "client"):
            thread_local.client = make_qdrant_client()
        return thread_local.client

    def _query(q, k):
        client = _get_client()
        return query_qdrant(client, collection, q, k, category_filter=category_filter, ef=ef)

    q_list = [q.tolist() for q in query_vecs]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_query, q, k) for q in q_list]
        for f in as_completed(futures):
            f.result()
    elapsed = time.perf_counter() - start
    return len(q_list) / elapsed


def measure_memory_qdrant(client, collection: str) -> float:
    """Return actual physical memory usage of Qdrant container in MB via Docker stats."""
    docker_mb = get_container_memory_usage_mb("qdrant-bench")
    if docker_mb is None:
        raise RuntimeError(
            "Could not read Qdrant container memory from Docker stats. "
            "Ensure the 'qdrant-bench' container is running."
        )
    return docker_mb


def fill_qdrant(client, collection: str, corpus: dict, dim: int = DIMENSIONS) -> tuple[float, float, int]:
    """Insert all corpus documents into Qdrant, then wait for HNSW build.
    Returns (ingest_time_s, index_build_time_s, total_inserted)."""
    from qdrant_client.models import PointStruct

    all_ids = sorted(corpus.keys())
    total = len(all_ids)
    point_id = 0

    start = time.perf_counter()

    # Create payload index on category for efficient filtered search
    client.create_payload_index(
        collection_name=collection,
        field_name="category",
        field_schema="keyword",
        wait=True,
    )

    print(f"  Inserting {total:,} documents …")
    for i in range(0, total, BATCH_SIZE):
        points = []
        for j in range(i, min(i + BATCH_SIZE, total)):
            did = all_ids[j]
            doc = corpus[did]
            text = f"{doc.get('title', '')}. {doc['text']}"
            emb = doc["embedding"] if isinstance(doc["embedding"], list) else doc["embedding"].tolist()
            points.append(PointStruct(
                id=point_id,
                vector=emb,
                payload={
                    "doc_id": did,
                    "text": text,
                    "category": f"cat_{j % NUM_CATEGORIES}",
                },
            ))
            point_id += 1
        client.upsert(collection_name=collection, points=points)
        if (i + BATCH_SIZE) % 100_000 < BATCH_SIZE:
            print(f"    {min(i + BATCH_SIZE, total):,} / {total:,} docs inserted")

    print(f"  Ingestion complete: {total:,} docs")
    ingest_time = time.perf_counter() - start

    # Wait for HNSW index to finish building
    print("  Waiting for Qdrant optimizer to finish …")
    idx_start = time.perf_counter()
    while True:
        info = client.get_collection(collection_name=collection)
        if info.optimizer_status == "ok" or str(info.optimizer_status.value) == "ok":
            break
        time.sleep(0.5)
    index_build_time = time.perf_counter() - idx_start

    docker_mb = get_container_memory_usage_mb("qdrant-bench")
    if docker_mb is not None:
        print(f"  Container memory: {docker_mb:.0f} MB / {CONTAINER_MEMORY_GB * 1024} MB")

    return ingest_time, index_build_time, total


def run_benchmark_qdrant(
    corpus: dict,
    query_vecs: np.ndarray,
    query_ids: list[str],
    qrels: dict,
    num_runs: int = NUM_RUNS,
    ckpt_db: dict | None = None,
    save_fn=None,
    hnsw_config: dict | None = None,
) -> BenchmarkResult:
    if ckpt_db is None:
        ckpt_db = {}
    if save_fn is None:
        save_fn = lambda: None

    if hnsw_config:
        db_label = hnsw_config["name"]
        ef_search = hnsw_config.get("ef_search")
        m = hnsw_config.get("m")
        ef_construct = hnsw_config.get("ef_construct")
    else:
        db_label = "Qdrant (Default)"
        ef_search = None
        m = None
        ef_construct = None

    print(f"\n{'='*60}")
    print(f"  {db_label}  |  {len(corpus):,} docs  |  index = HNSW  |  protocol = gRPC")
    print(f"{'='*60}")

    # --- Ingest ---
    ingest_ckpt = ckpt_db.get("ingest")
    client = None
    collection = "documents"

    if ingest_ckpt:
        try:
            wait_for_qdrant()
            client = make_qdrant_client()
            info = client.get_collection(collection_name=collection)
            count = info.points_count
            if count == ingest_ckpt["dataset_size"]:
                print(f"  Resuming — {count:,} docs already ingested (skipping ingest)")
                ingest_s = ingest_ckpt["ingest_time_s"]
                idx_build_s = ingest_ckpt["index_build_time_s"]
                n = ingest_ckpt["dataset_size"]
                mem_mb = ingest_ckpt["memory_mb"]
            else:
                ingest_ckpt = None
        except Exception:
            ingest_ckpt = None

    if not ingest_ckpt:
        client, collection = setup_qdrant(DIMENSIONS, m=m, ef_construct=ef_construct)
        ingest_s, idx_build_s, n = fill_qdrant(client, collection, corpus)
        print(f"  Ingestion done in {ingest_s:.2f}s — {n:,} documents")
        print(f"  Index built in {idx_build_s:.2f}s")
        mem_mb = measure_memory_qdrant(client, collection)
        print(f"  Memory usage: {mem_mb:.1f} MB")
        ckpt_db["ingest"] = {
            "ingest_time_s": ingest_s, "index_build_time_s": idx_build_s,
            "dataset_size": n, "memory_mb": mem_mb,
        }
        save_fn()

    # --- Measurement runs ---
    completed_runs = ckpt_db.get("runs", [])
    all_p50 = [r["p50"] for r in completed_runs]
    all_p95 = [r["p95"] for r in completed_runs]
    all_p99 = [r["p99"] for r in completed_runs]
    all_qps = [r["qps"] for r in completed_runs]
    all_recall = {k: [] for k in TOP_K_VALUES}
    for r in completed_runs:
        for k in TOP_K_VALUES:
            all_recall[k].append(r["recall"][str(k)])
    all_fp50 = [r["fp50"] for r in completed_runs]
    all_fp95 = [r["fp95"] for r in completed_runs]

    start_run = len(completed_runs) + 1
    if start_run > 1:
        print(f"\n  Resuming from run {start_run} ({start_run - 1} runs already saved)")

    query_fn = lambda q, k: query_qdrant(client, collection, q, k, ef=ef_search)
    max_k = max(TOP_K_VALUES)

    for run in range(start_run, num_runs + 1):
        print(f"\n  --- Run {run}/{num_runs} ---")

        # Wait for optimizer to be idle before measuring
        print("  Waiting for optimizer to settle …")
        while True:
            info = client.get_collection(collection_name=collection)
            if info.optimizer_status == "ok" or str(info.optimizer_status.value) == "ok":
                break
            time.sleep(0.5)

        print("  Measuring query duration …")
        lats = measure_durations(query_fn, query_vecs, k=10)
        p50 = float(np.percentile(lats, 50))
        p95 = float(np.percentile(lats, 95))
        p99 = float(np.percentile(lats, 99))
        all_p50.append(p50)
        all_p95.append(p95)
        all_p99.append(p99)

        print(f"  Measuring throughput ({CONCURRENT_WORKERS} workers) …")
        qps = measure_throughput_qdrant(collection, query_vecs, k=10, ef=ef_search)
        all_qps.append(qps)

        print("  Computing recall@k (vs human relevance judgments) …")
        retrieved_max_k = [
            query_qdrant(client, collection, q.tolist(), max_k, ef=ef_search)
            for q in query_vecs
        ]
        run_recall = {}
        for k in TOP_K_VALUES:
            recall = compute_recall_qrels(retrieved_max_k, query_ids, qrels, k)
            all_recall[k].append(recall)
            run_recall[str(k)] = recall
            print(f"    recall@{k} = {recall:.4f}")

        print("  Measuring filtered search duration …")
        f_lats = measure_durations(
            lambda q, k: query_qdrant(
                client, collection, q, k, category_filter="cat_0", ef=ef_search,
            ),
            query_vecs, k=10,
        )
        fp50 = float(np.percentile(f_lats, 50))
        fp95 = float(np.percentile(f_lats, 95))
        all_fp50.append(fp50)
        all_fp95.append(fp95)

        if "runs" not in ckpt_db:
            ckpt_db["runs"] = []
        ckpt_db["runs"].append({
            "p50": p50, "p95": p95, "p99": p99,
            "qps": qps, "recall": run_recall,
            "fp50": fp50, "fp95": fp95,
        })
        save_fn()

    return BenchmarkResult(
        db_name=db_label,
        dataset_size=n,
        ingest_time_s=ingest_s,
        index_build_time_s=idx_build_s,
        duration_p50_ms=statistics.mean(all_p50),
        duration_p50_std=statistics.stdev(all_p50) if len(all_p50) > 1 else 0.0,
        duration_p95_ms=statistics.mean(all_p95),
        duration_p95_std=statistics.stdev(all_p95) if len(all_p95) > 1 else 0.0,
        duration_p99_ms=statistics.mean(all_p99),
        duration_p99_std=statistics.stdev(all_p99) if len(all_p99) > 1 else 0.0,
        throughput_qps=statistics.mean(all_qps),
        throughput_qps_std=statistics.stdev(all_qps) if len(all_qps) > 1 else 0.0,
        recall_at_k={k: statistics.mean(v) for k, v in all_recall.items()},
        recall_at_k_std={k: (statistics.stdev(v) if len(v) > 1 else 0.0) for k, v in all_recall.items()},
        filtered_duration_p50_ms=statistics.mean(all_fp50),
        filtered_duration_p50_std=statistics.stdev(all_fp50) if len(all_fp50) > 1 else 0.0,
        filtered_duration_p95_ms=statistics.mean(all_fp95),
        filtered_duration_p95_std=statistics.stdev(all_fp95) if len(all_fp95) > 1 else 0.0,
        memory_mb=mem_mb,
        num_runs=num_runs,
    )
