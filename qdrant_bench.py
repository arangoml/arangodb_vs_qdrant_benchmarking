"""
Qdrant setup, querying, ingestion, and benchmark runner.
"""

import time
import hashlib
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np

from config import (
    QDRANT_HOST, QDRANT_HTTP_PORT, QDRANT_GRPC_PORT,
    DIMENSIONS, TOP_K_VALUES, CONTAINER_MEMORY_GB, TARGET_DOC_COUNT,
    FILL_BATCH_SIZE, CONCURRENT_WORKERS, NUM_CATEGORIES, NUM_RUNS,
    EF_SWEEP,
)
from docker import get_container_memory_usage_mb
from measure import (
    BenchmarkResult, RecallDurationPoint,
    measure_durations, compute_recall_qrels,
    generate_synthetic_batch, precompute_corpus_arrays,
)


def make_qdrant_client():
    """Create a new Qdrant gRPC client (one per thread for thread safety)."""
    from qdrant_client import QdrantClient
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_HTTP_PORT, grpc_port=QDRANT_GRPC_PORT, prefer_grpc=True)


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


def setup_qdrant(dim: int):
    from qdrant_client.models import Distance, VectorParams

    wait_for_qdrant()
    client = make_qdrant_client()
    collection = "documents"

    if client.collection_exists(collection):
        client.delete_collection(collection)

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
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
        return query_qdrant(client, collection, q, k, category_filter=category_filter)

    q_list = [q.tolist() for q in query_vecs]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_query, q, k) for q in q_list]
        for f in as_completed(futures):
            f.result()
    elapsed = time.perf_counter() - start
    return len(q_list) / elapsed


def measure_memory_qdrant(client, collection: str) -> float:
    """Return actual memory usage of the Qdrant collection in MB via telemetry API."""
    try:
        resp = requests.get(
            f"http://{QDRANT_HOST}:{QDRANT_HTTP_PORT}/telemetry?details_level=4",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        total_ram = 0
        collections = data.get("result", {}).get("collections", {}).get("collections", [])
        for col in collections:
            col_name = col.get("id", "")
            if col_name != collection:
                continue
            for shard in col.get("shards", []):
                for segment in shard.get("local", {}).get("segments", []):
                    info = segment.get("info", {})
                    total_ram += info.get("ram_usage_bytes", 0)
        if total_ram > 0:
            return total_ram / (1024 * 1024)
    except Exception as e:
        print(f"  Warning: telemetry API failed ({e}), falling back to estimate")
    # Fallback: estimate from point count
    info = client.get_collection(collection_name=collection)
    points_count = info.points_count or 0
    dim = info.config.params.vectors.size if info.config.params.vectors else DIMENSIONS
    vector_bytes = points_count * dim * 4
    return vector_bytes * 2.5 / (1024 * 1024)


def fill_qdrant_fixed(
    client, collection: str, corpus: dict, target_docs: int,
    dim: int = DIMENSIONS,
) -> tuple[float, int]:
    """Insert exactly target_docs documents into Qdrant, then wait for HNSW build."""
    from qdrant_client.models import PointStruct

    all_ids, corpus_embeddings, corpus_texts = precompute_corpus_arrays(corpus)
    rng = np.random.default_rng(42)
    total_inserted = 0
    point_id = 0
    subbatch_size = 500

    start = time.perf_counter()

    # Create payload index on category for efficient filtered search
    client.create_payload_index(
        collection_name=collection,
        field_name="category",
        field_schema="keyword",
        wait=True,
    )

    # Phase 1: Insert real corpus documents
    real_count = min(len(all_ids), target_docs)
    print(f"  Phase 1: Inserting {real_count:,} real corpus documents …")
    for i in range(0, real_count, subbatch_size):
        points = []
        for j in range(i, min(i + subbatch_size, real_count)):
            points.append(PointStruct(
                id=point_id,
                vector=corpus_embeddings[j].tolist(),
                payload={
                    "doc_id": all_ids[j],
                    "text": corpus_texts[j],
                    "category": f"cat_{int(hashlib.md5(all_ids[j].encode()).hexdigest(), 16) % NUM_CATEGORIES}",
                },
            ))
            point_id += 1
        client.upsert(collection_name=collection, points=points)
        total_inserted += len(points)
    print(f"  Phase 1 done: {total_inserted:,} docs inserted")

    # Phase 2: Fill remainder with synthetic docs
    remaining = target_docs - total_inserted
    if remaining > 0:
        print(f"  Phase 2: Inserting {remaining:,} synthetic docs …")
        syn_index = 0
        while total_inserted < target_docs:
            batch_size = min(FILL_BATCH_SIZE, target_docs - total_inserted)
            embs, doc_ids, texts, categories = generate_synthetic_batch(
                corpus_embeddings, corpus_texts, rng, syn_index, batch_size, dim,
            )
            syn_index += batch_size

            for i in range(0, batch_size, subbatch_size):
                sub_end = min(i + subbatch_size, batch_size)
                points = []
                for j in range(i, sub_end):
                    points.append(PointStruct(
                        id=point_id,
                        vector=embs[j].tolist(),
                        payload={
                            "doc_id": doc_ids[j],
                            "text": texts[j],
                            "category": categories[j],
                        },
                    ))
                    point_id += 1
                client.upsert(collection_name=collection, points=points)
                total_inserted += len(points)
            print(f"  {total_inserted:,} / {target_docs:,} docs inserted")

    print(f"  Ingestion complete: {total_inserted:,} docs")

    # Wait for HNSW index to finish building
    print("  Waiting for Qdrant optimizer to finish …")
    while True:
        info = client.get_collection(collection_name=collection)
        if info.optimizer_status == "ok" or str(info.optimizer_status.value) == "ok":
            break
        time.sleep(0.5)

    docker_mb = get_container_memory_usage_mb("qdrant-bench")
    if docker_mb is not None:
        print(f"  Container memory: {docker_mb:.0f} MB / {CONTAINER_MEMORY_GB * 1024} MB")

    elapsed = time.perf_counter() - start
    return elapsed, total_inserted



def run_benchmark_qdrant(
    corpus: dict,
    query_vecs: np.ndarray,
    query_ids: list[str],
    qrels: dict,
    num_runs: int = NUM_RUNS,
    ckpt_db: dict | None = None,
    save_fn=None,
) -> tuple[BenchmarkResult, list[RecallDurationPoint]]:
    ckpt_db = ckpt_db or {}
    if save_fn is None:
        save_fn = lambda: None

    print(f"\n{'='*60}")
    print(f"  Qdrant    |  {TARGET_DOC_COUNT:,} docs  |  index = HNSW  |  protocol = gRPC")
    print(f"{'='*60}")

    # --- Ingest (skip if checkpoint has it and data is still in the container) ---
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
                idx_time = ingest_ckpt["index_time_s"]
                n = ingest_ckpt["dataset_size"]
                mem_mb = ingest_ckpt["memory_mb"]
            else:
                ingest_ckpt = None
        except Exception:
            ingest_ckpt = None

    if not ingest_ckpt:
        client, collection = setup_qdrant(DIMENSIONS)
        print(f"  Inserting {TARGET_DOC_COUNT:,} docs …")
        idx_time, n = fill_qdrant_fixed(client, collection, corpus, TARGET_DOC_COUNT)
        print(f"  Ingestion done in {idx_time:.2f}s — {n:,} documents")
        mem_mb = measure_memory_qdrant(client, collection)
        print(f"  Memory usage (est.): {mem_mb:.1f} MB")
        ckpt_db["ingest"] = {"index_time_s": idx_time, "dataset_size": n, "memory_mb": mem_mb}
        save_fn()

    # --- Measurement runs (skip already-completed runs) ---
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

    query_fn = lambda q, k: query_qdrant(client, collection, q, k)
    max_k = max(TOP_K_VALUES)

    for run in range(start_run, num_runs + 1):
        print(f"\n  --- Run {run}/{num_runs} ---")

        print("  Measuring query duration …")
        lats = measure_durations(query_fn, query_vecs, k=10)
        p50 = float(np.percentile(lats, 50))
        p95 = float(np.percentile(lats, 95))
        p99 = float(np.percentile(lats, 99))
        all_p50.append(p50)
        all_p95.append(p95)
        all_p99.append(p99)

        print(f"  Measuring throughput ({CONCURRENT_WORKERS} workers) …")
        qps = measure_throughput_qdrant(collection, query_vecs, k=10)
        all_qps.append(qps)

        print("  Computing recall@k (vs human relevance judgments) …")
        retrieved_max_k = [query_qdrant(client, collection, q.tolist(), max_k) for q in query_vecs]
        run_recall = {}
        for k in TOP_K_VALUES:
            recall = compute_recall_qrels(retrieved_max_k, query_ids, qrels, k)
            all_recall[k].append(recall)
            run_recall[str(k)] = recall
            print(f"    recall@{k} = {recall:.4f}")

        print("  Measuring filtered search duration …")
        f_lats = measure_durations(
            lambda q, k: query_qdrant(client, collection, q, k, category_filter="cat_0"),
            query_vecs, k=10,
        )
        fp50 = float(np.percentile(f_lats, 50))
        fp95 = float(np.percentile(f_lats, 95))
        all_fp50.append(fp50)
        all_fp95.append(fp95)

        # Save checkpoint after each run
        if "runs" not in ckpt_db:
            ckpt_db["runs"] = []
        ckpt_db["runs"].append({
            "p50": p50, "p95": p95, "p99": p99,
            "qps": qps, "recall": run_recall,
            "fp50": fp50, "fp95": fp95,
        })
        save_fn()

    # --- Pareto sweep (skip already-completed points) ---
    completed_pareto = ckpt_db.get("pareto", [])
    completed_efs = {p["param_value"] for p in completed_pareto}

    sweep_values = sorted(EF_SWEEP)
    remaining_sweep = [v for v in sweep_values if v not in completed_efs]

    pareto_points = [
        RecallDurationPoint(
            db_name="Qdrant", dataset_size=n,
            param_name="ef", param_value=p["param_value"],
            recall_at_10=p["recall_at_10"], duration_p50_ms=p["p50"], duration_p95_ms=p["p95"],
        )
        for p in completed_pareto
    ]

    if remaining_sweep:
        if completed_pareto:
            print(f"\n  Resuming pareto sweep — {len(completed_pareto)} points already saved")
        print(f"\n  Sweeping ef: {remaining_sweep}")
        for ef_val in remaining_sweep:
            sweep_query_fn = lambda q, k, _ef=ef_val: query_qdrant(client, collection, q, k, ef=_ef)

            for q in query_vecs[:min(10, len(query_vecs))]:
                sweep_query_fn(q.tolist(), 10)

            lats = []
            for q in query_vecs:
                t0 = time.perf_counter()
                sweep_query_fn(q.tolist(), 10)
                lats.append((time.perf_counter() - t0) * 1000)

            retrieved = [sweep_query_fn(q.tolist(), 50) for q in query_vecs]
            recall = compute_recall_qrels(retrieved, query_ids, qrels, 10)

            p50 = float(np.percentile(lats, 50))
            p95 = float(np.percentile(lats, 95))
            print(f"    ef={ef_val:>4d}  →  recall@10={recall:.4f}  p50={p50:.2f}ms  p95={p95:.2f}ms")

            point = RecallDurationPoint(
                db_name="Qdrant", dataset_size=n,
                param_name="ef", param_value=ef_val,
                recall_at_10=recall, duration_p50_ms=p50, duration_p95_ms=p95,
            )
            pareto_points.append(point)

            if "pareto" not in ckpt_db:
                ckpt_db["pareto"] = []
            ckpt_db["pareto"].append({
                "param_value": ef_val, "recall_at_10": recall, "p50": p50, "p95": p95,
            })
            save_fn()

    return BenchmarkResult(
        db_name="Qdrant",
        dataset_size=n,
        index_time_s=idx_time,
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
    ), pareto_points
