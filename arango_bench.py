"""
ArangoDB setup, querying, ingestion, and benchmark runner.
"""

import time
import hashlib
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np

from config import (
    ARANGO_HOST, ARANGO_DB, ARANGO_USER, ARANGO_PASS, ARANGO_COLLECTION,
    DIMENSIONS, TOP_K_VALUES, CONTAINER_MEMORY_GB, TARGET_DOC_COUNT,
    FILL_BATCH_SIZE, CONCURRENT_WORKERS, NUM_CATEGORIES, NUM_RUNS,
    NPROBE_SWEEP,
)
from docker import get_container_memory_usage_mb
from measure import (
    BenchmarkResult, RecallDurationPoint,
    measure_durations, compute_recall_qrels,
    generate_synthetic_batch, precompute_corpus_arrays,
)


def wait_for_arango(timeout: int = 120):
    """Block until ArangoDB is reachable and ready (handles container restarts)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(
                f"{ARANGO_HOST}/_api/version",
                auth=(ARANGO_USER, ARANGO_PASS),
                timeout=2,
            )
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"ArangoDB not reachable at {ARANGO_HOST} after {timeout}s")


def setup_arango(dim: int):
    from arango import ArangoClient
    wait_for_arango()
    client = ArangoClient(hosts=ARANGO_HOST, request_timeout=600)
    sys_db = client.db("_system", username=ARANGO_USER, password=ARANGO_PASS)
    if sys_db.has_database(ARANGO_DB):
        sys_db.delete_database(ARANGO_DB)
    sys_db.create_database(ARANGO_DB)
    db = client.db(ARANGO_DB, username=ARANGO_USER, password=ARANGO_PASS)
    for _ in range(30):
        try:
            db.properties()
            break
        except Exception:
            time.sleep(1)
    if db.has_collection(ARANGO_COLLECTION):
        db.delete_collection(ARANGO_COLLECTION)
    col = db.create_collection(ARANGO_COLLECTION)
    # Verify collection is ready for writes with a test document
    for attempt in range(30):
        try:
            col.insert({"_key": "_healthcheck", "test": True})
            col.delete("_healthcheck")
            break
        except Exception:
            time.sleep(1)
    return db, col


def query_arango(db, query_vec: list[float], k: int,
                 category_filter: str | None = None) -> list[str]:
    """Run an ANN vector query via AQL using the IVF index. Returns list of doc_id."""
    bind = {"@col": ARANGO_COLLECTION, "qvec": query_vec, "k": k}

    if category_filter:
        aql = """
        FOR doc IN @@col
            FILTER doc.category == @cat
            LET sim = APPROX_NEAR_COSINE(doc.embedding, @qvec)
            SORT sim DESC
            LIMIT @k
            RETURN doc.doc_id
        """
        bind["cat"] = category_filter
    else:
        aql = """
        FOR doc IN @@col
            LET sim = APPROX_NEAR_COSINE(doc.embedding, @qvec)
            SORT sim DESC
            LIMIT @k
            RETURN doc.doc_id
        """

    cursor = db.aql.execute(aql, bind_vars=bind, count=True)
    return list(cursor)


def measure_throughput_arango(query_vecs: np.ndarray, k: int,
                              category_filter: str | None = None,
                              workers: int = CONCURRENT_WORKERS) -> float:
    """Run ArangoDB queries concurrently with per-thread DB connections."""
    import threading
    from arango import ArangoClient

    thread_local = threading.local()

    def _get_db():
        if not hasattr(thread_local, "db"):
            client = ArangoClient(hosts=ARANGO_HOST, request_timeout=600)
            thread_local.db = client.db(ARANGO_DB, username=ARANGO_USER, password=ARANGO_PASS)
        return thread_local.db

    def _query(q, k):
        db = _get_db()
        return query_arango(db, q, k, category_filter=category_filter)

    q_list = [q.tolist() for q in query_vecs]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_query, q, k) for q in q_list]
        for f in as_completed(futures):
            f.result()
    elapsed = time.perf_counter() - start
    return len(q_list) / elapsed


def measure_memory_arango(db) -> float:
    """Return memory usage of ArangoDB in MB."""
    col = db.collection(ARANGO_COLLECTION)
    stats = col.statistics()
    figures = stats.get("figures", {})
    indexes_size = figures.get("indexes", {}).get("size", 0)
    documents_size = figures.get("documents", {}).get("size", 0)
    api_mb = (indexes_size + documents_size) / (1024 * 1024)
    if api_mb > 1:
        return api_mb
    docker_mb = get_container_memory_usage_mb("arango-bench")
    if docker_mb is not None:
        return docker_mb
    return api_mb


def _insert_subbatches(col, docs: list[dict], subbatch_size: int = 500):
    """Insert a large list of docs into ArangoDB in sub-batches with retry."""
    for i in range(0, len(docs), subbatch_size):
        for attempt in range(3):
            try:
                col.insert_many(docs[i:i + subbatch_size], overwrite=True)
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  Insert failed ({e}), retrying in 5s …")
                    time.sleep(5)
                else:
                    raise


def fill_arango_fixed(
    db, col, corpus: dict, target_docs: int,
    dim: int = DIMENSIONS,
) -> tuple[float, int]:
    """Insert exactly target_docs documents into ArangoDB, then build the IVF index."""
    all_ids, corpus_embeddings, corpus_texts = precompute_corpus_arrays(corpus)
    rng = np.random.default_rng(42)
    total_inserted = 0

    start = time.perf_counter()

    # Phase 1: Insert real corpus documents
    real_count = min(len(all_ids), target_docs)
    print(f"  Phase 1: Inserting {real_count:,} real corpus documents …")
    batch = []
    for i in range(real_count):
        did = all_ids[i]
        batch.append({
            "_key": did.replace("/", "_"),
            "doc_id": did,
            "text": corpus_texts[i],
            "category": f"cat_{int(hashlib.md5(did.encode()).hexdigest(), 16) % NUM_CATEGORIES}",
            "embedding": corpus_embeddings[i].tolist(),
        })
    _insert_subbatches(col, batch)
    total_inserted = len(batch)
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
            batch = []
            for i in range(batch_size):
                batch.append({
                    "_key": doc_ids[i],
                    "doc_id": doc_ids[i],
                    "text": texts[i],
                    "category": categories[i],
                    "embedding": embs[i].tolist(),
                })
            _insert_subbatches(col, batch)
            total_inserted += len(batch)
            print(f"  {total_inserted:,} / {target_docs:,} docs inserted")

    print(f"  Ingestion complete: {total_inserted:,} docs")

    # Build IVF vector index
    n_lists = max(1, int(total_inserted ** 0.5))
    print(f"  Creating vector index (nLists={n_lists}) over {total_inserted:,} docs …")
    col.add_index({
        "type": "vector",
        "name": "vector_cosine",
        "fields": ["embedding"],
        "storedValues": ["category"],
        "params": {
            "metric": "cosine",
            "dimension": dim,
            "nLists": n_lists,
            "defaultNProbe": max(1, n_lists // 4),
            "trainingIterations": 25,
        },
    })

    docker_mb = get_container_memory_usage_mb("arango-bench")
    if docker_mb is not None:
        print(f"  Container memory: {docker_mb:.0f} MB / {CONTAINER_MEMORY_GB * 1024} MB")

    elapsed = time.perf_counter() - start
    return elapsed, total_inserted



def run_benchmark_arango(
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
    print(f"  ArangoDB  |  {TARGET_DOC_COUNT:,} docs  |  index = IVF  |  protocol = HTTP")
    print(f"{'='*60}")

    # --- Ingest (skip if checkpoint has it and data is still in the container) ---
    ingest_ckpt = ckpt_db.get("ingest")
    db = None

    if ingest_ckpt:
        from arango import ArangoClient
        try:
            wait_for_arango()
            client = ArangoClient(hosts=ARANGO_HOST, request_timeout=600)
            db = client.db(ARANGO_DB, username=ARANGO_USER, password=ARANGO_PASS)
            col = db.collection(ARANGO_COLLECTION)
            count = col.count()
            if count == ingest_ckpt["dataset_size"]:
                print(f"  Resuming — {count:,} docs already ingested (skipping ingest)")
                idx_time = ingest_ckpt["index_time_s"]
                n = ingest_ckpt["dataset_size"]
                mem_mb = ingest_ckpt["memory_mb"]
            else:
                ingest_ckpt = None  # count mismatch, redo
        except Exception:
            ingest_ckpt = None  # DB not reachable or missing, redo

    if not ingest_ckpt:
        db, col = setup_arango(DIMENSIONS)
        print(f"  Inserting {TARGET_DOC_COUNT:,} docs …")
        idx_time, n = fill_arango_fixed(db, col, corpus, TARGET_DOC_COUNT)
        print(f"  Ingestion done in {idx_time:.2f}s — {n:,} documents")
        mem_mb = measure_memory_arango(db)
        print(f"  Memory usage: {mem_mb:.1f} MB")
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

    query_fn = lambda q, k: query_arango(db, q, k)
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
        qps = measure_throughput_arango(query_vecs, k=10)
        all_qps.append(qps)

        print("  Computing recall@k (vs human relevance judgments) …")
        retrieved_max_k = [query_arango(db, q.tolist(), max_k) for q in query_vecs]
        run_recall = {}
        for k in TOP_K_VALUES:
            recall = compute_recall_qrels(retrieved_max_k, query_ids, qrels, k)
            all_recall[k].append(recall)
            run_recall[str(k)] = recall
            print(f"    recall@{k} = {recall:.4f}")

        print("  Measuring filtered search duration …")
        f_lats = measure_durations(
            lambda q, k: query_arango(db, q, k, category_filter="cat_0"),
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
    completed_nprobes = {p["param_value"] for p in completed_pareto}

    n_lists = max(1, int(n ** 0.5))
    sweep_values = sorted(set(v for v in NPROBE_SWEEP if v <= n_lists))

    remaining_sweep = [v for v in sweep_values if v not in completed_nprobes]
    pareto_points = [
        RecallDurationPoint(
            db_name="ArangoDB", dataset_size=n,
            param_name="nProbe", param_value=p["param_value"],
            recall_at_10=p["recall_at_10"], duration_p50_ms=p["p50"], duration_p95_ms=p["p95"],
        )
        for p in completed_pareto
    ]

    if remaining_sweep:
        if completed_pareto:
            print(f"\n  Resuming pareto sweep — {len(completed_pareto)} points already saved")
        print(f"\n  Sweeping nProbe (index rebuild per value): {remaining_sweep}")
        col = db.collection(ARANGO_COLLECTION)
        for n_probe in remaining_sweep:
            try:
                col.delete_index("vector_cosine")
            except Exception:
                pass
            col.add_index({
                "type": "vector",
                "name": "vector_cosine",
                "fields": ["embedding"],
                "storedValues": ["category"],
                "params": {
                    "metric": "cosine",
                    "dimension": DIMENSIONS,
                    "nLists": n_lists,
                    "defaultNProbe": n_probe,
                    "trainingIterations": 25,
                },
            })

            sweep_query_fn = lambda q, k: query_arango(db, q, k)
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
            print(f"    nProbe={n_probe:>4d}  →  recall@10={recall:.4f}  p50={p50:.2f}ms  p95={p95:.2f}ms")

            point = RecallDurationPoint(
                db_name="ArangoDB", dataset_size=n,
                param_name="nProbe", param_value=n_probe,
                recall_at_10=recall, duration_p50_ms=p50, duration_p95_ms=p95,
            )
            pareto_points.append(point)

            if "pareto" not in ckpt_db:
                ckpt_db["pareto"] = []
            ckpt_db["pareto"].append({
                "param_value": n_probe, "recall_at_10": recall, "p50": p50, "p95": p95,
            })
            save_fn()

    return BenchmarkResult(
        db_name="ArangoDB",
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
