"""
ArangoDB setup, querying, ingestion, and benchmark runner.
"""

import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np

from config import (
    ARANGO_HOST, ARANGO_DB, ARANGO_USER, ARANGO_PASS, ARANGO_COLLECTION,
    DIMENSIONS, TOP_K_VALUES, PRIMARY_RECALL_K, CONTAINER_MEMORY_GB,
    BATCH_SIZE, CONCURRENT_WORKERS, NUM_CATEGORIES, NUM_RUNS,
)
from docker import get_container_memory_usage_mb
from measure import (
    BenchmarkResult,
    measure_durations, compute_recall_qrels, compute_topk_metrics_qrels,
    has_topk_metric_series,
)


def wait_for_arango(timeout: int = 120):
    """Block until ArangoDB is reachable and past its Docker init-restart cycle.

    The ArangoDB Docker entrypoint passes *all* CMD args (including
    ``--server.endpoint tcp://0.0.0.0:8529``) to a temporary init arangod.
    That means the init instance is reachable on port 8529, but it gets
    SIGTERM'd once init is done and a fresh arangod takes its place.

    We handle this by waiting for the API to come up, then watching for the
    expected down/up transition.  If the API never goes down (volumes already
    initialised → no init step), we proceed after a short grace period.
    """
    GRACE_SECONDS = 15  # max time to wait for the init-restart
    deadline = time.time() + timeout

    # Phase 1: wait for the API to respond at all
    while time.time() < deadline:
        try:
            r = requests.get(
                f"{ARANGO_HOST}/_api/version",
                auth=(ARANGO_USER, ARANGO_PASS),
                timeout=2,
            )
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        raise RuntimeError(f"ArangoDB not reachable at {ARANGO_HOST} after {timeout}s")

    # Phase 2: watch for the init-restart (API goes down after init SIGTERM)
    grace_deadline = min(time.time() + GRACE_SECONDS, deadline)
    saw_down = False
    while time.time() < grace_deadline:
        try:
            r = requests.get(
                f"{ARANGO_HOST}/_api/version",
                auth=(ARANGO_USER, ARANGO_PASS),
                timeout=2,
            )
            if r.status_code != 200:
                saw_down = True
                break
        except Exception:
            saw_down = True
            break
        time.sleep(1)

    if not saw_down:
        return  # already initialised, no restart happened

    # Phase 3: wait for the real arangod to come up
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
    client = ArangoClient(hosts=ARANGO_HOST, request_timeout=3600)
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
    for attempt in range(30):
        try:
            col.insert({"_key": "_healthcheck", "test": True})
            col.delete("_healthcheck")
            break
        except Exception:
            time.sleep(1)
    return db, col


def query_arango(db, query_vec: list[float], k: int,
                 category_filter: str | None = None,
                 n_probe: int | None = None) -> list[str]:
    """Run an ANN vector query via AQL using the IVF index. Returns list of doc_id."""
    bind = {"@col": ARANGO_COLLECTION, "qvec": query_vec, "k": k}

    opts = ", { nProbe: @nProbe }" if n_probe is not None else ""
    if n_probe is not None:
        bind["nProbe"] = n_probe

    if category_filter:
        aql = f"""
        FOR doc IN @@col
            FILTER doc.category == @cat
            LET sim = APPROX_NEAR_COSINE(doc.embedding, @qvec{opts})
            SORT sim DESC
            LIMIT @k
            RETURN doc.doc_id
        """
        bind["cat"] = category_filter
    else:
        aql = f"""
        FOR doc IN @@col
            LET sim = APPROX_NEAR_COSINE(doc.embedding, @qvec{opts})
            SORT sim DESC
            LIMIT @k
            RETURN doc.doc_id
        """

    cursor = db.aql.execute(aql, bind_vars=bind)
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
            client = ArangoClient(hosts=ARANGO_HOST, request_timeout=3600)
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
    """Return actual physical memory usage of ArangoDB container in MB via Docker stats."""
    docker_mb = get_container_memory_usage_mb("arango-bench")
    if docker_mb is None:
        raise RuntimeError(
            "Could not read ArangoDB container memory from Docker stats. "
            "Ensure the 'arango-bench' container is running."
        )
    return docker_mb


def _insert_subbatches(col, docs: list[dict], subbatch_size: int = BATCH_SIZE):
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


def fill_arango(db, col, corpus: dict, dim: int = DIMENSIONS) -> tuple[float, float, int]:
    """Insert all corpus documents into ArangoDB, then build the IVF index.
    Returns (ingest_time_s, index_build_time_s, total_inserted)."""
    all_ids = sorted(corpus.keys())
    total = len(all_ids)

    start = time.perf_counter()

    print(f"  Inserting {total:,} documents …")
    batch = []
    for i, did in enumerate(all_ids):
        doc = corpus[did]
        text = f"{doc.get('title', '')}. {doc['text']}"
        batch.append({
            "_key": did.replace("/", "_"),
            "doc_id": did,
            "text": text,
            "category": f"cat_{i % NUM_CATEGORIES}",
            "embedding": doc["embedding"] if isinstance(doc["embedding"], list) else doc["embedding"].tolist(),
        })
        if len(batch) >= BATCH_SIZE:
            _insert_subbatches(col, batch)
            batch = []
            if (i + 1) % 100_000 == 0:
                print(f"    {i + 1:,} / {total:,} docs inserted")
    if batch:
        _insert_subbatches(col, batch)
    print(f"  Ingestion complete: {total:,} docs")
    ingest_time = time.perf_counter() - start

    # Build IVF vector index
    n_lists = max(1, int(15 * total ** 0.5))
    print(f"  Creating vector index (nLists={n_lists}) over {total:,} docs …")
    idx_start = time.perf_counter()
    col.add_index({
        "type": "vector",
        "name": "vector_cosine",
        "fields": ["embedding"],
        "storedValues": ["category"],
        "params": {
            "metric": "cosine",
            "dimension": dim,
            "nLists": n_lists,
            "defaultNProbe": max(1, int(n_lists ** 0.5)),
            "trainingIterations": 25,
        },
    })
    index_build_time = time.perf_counter() - idx_start

    docker_mb = get_container_memory_usage_mb("arango-bench")
    if docker_mb is not None:
        print(f"  Container memory: {docker_mb:.0f} MB / {CONTAINER_MEMORY_GB * 1024} MB")

    return ingest_time, index_build_time, total


def run_benchmark_arango(
    corpus: dict,
    query_vecs: np.ndarray,
    query_ids: list[str],
    qrels: dict,
    num_runs: int = NUM_RUNS,
    ckpt_db: dict | None = None,
    save_fn=None,
) -> BenchmarkResult:
    if ckpt_db is None:
        ckpt_db = {}
    if save_fn is None:
        save_fn = lambda: None

    print(f"\n{'='*60}")
    print(f"  ArangoDB  |  {len(corpus):,} docs  |  index = IVF  |  protocol = HTTP")
    print(f"{'='*60}")

    # --- Ingest (skip if checkpoint has it and data is still in the container) ---
    ingest_ckpt = ckpt_db.get("ingest")
    db = None

    if ingest_ckpt:
        from arango import ArangoClient
        try:
            wait_for_arango()
            client = ArangoClient(hosts=ARANGO_HOST, request_timeout=3600)
            db = client.db(ARANGO_DB, username=ARANGO_USER, password=ARANGO_PASS)
            col = db.collection(ARANGO_COLLECTION)
            count = col.count()
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
        db, col = setup_arango(DIMENSIONS)
        ingest_s, idx_build_s, n = fill_arango(db, col, corpus)
        print(f"  Ingestion done in {ingest_s:.2f}s — {n:,} documents")
        print(f"  Index built in {idx_build_s:.2f}s")
        mem_mb = measure_memory_arango(db)
        print(f"  Memory usage: {mem_mb:.1f} MB")
        ckpt_db["ingest"] = {
            "ingest_time_s": ingest_s, "index_build_time_s": idx_build_s,
            "dataset_size": n, "memory_mb": mem_mb,
        }
        save_fn()

    # --- Measurement runs ---
    completed_runs = [r for r in ckpt_db.get("runs", []) if has_topk_metric_series(r)]
    if len(completed_runs) != len(ckpt_db.get("runs", [])):
        ckpt_db["runs"] = completed_runs
        save_fn()
    all_p50 = [r["p50"] for r in completed_runs]
    all_p95 = [r["p95"] for r in completed_runs]
    all_p99 = [r["p99"] for r in completed_runs]
    all_qps = [r["qps"] for r in completed_runs]
    all_recall = {k: [] for k in TOP_K_VALUES}
    all_precision = {k: [] for k in TOP_K_VALUES}
    all_ndcg = {k: [] for k in TOP_K_VALUES}
    all_success = {k: [] for k in TOP_K_VALUES}
    for r in completed_runs:
        for k in TOP_K_VALUES:
            all_recall[k].append(r["recall"][str(k)])
            all_precision[k].append(r["topk_metrics"]["precision"][str(k)])
            all_ndcg[k].append(r["topk_metrics"]["ndcg"][str(k)])
            all_success[k].append(r["topk_metrics"]["success"][str(k)])
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
        run_topk_metrics = {"precision": {}, "ndcg": {}, "success": {}}
        for k in TOP_K_VALUES:
            topk_metrics = compute_topk_metrics_qrels(
                retrieved_max_k, query_ids, qrels, k=k,
            )
            all_precision[k].append(topk_metrics["precision"])
            all_ndcg[k].append(topk_metrics["ndcg"])
            all_success[k].append(topk_metrics["success"])
            run_topk_metrics["precision"][str(k)] = topk_metrics["precision"]
            run_topk_metrics["ndcg"][str(k)] = topk_metrics["ndcg"]
            run_topk_metrics["success"][str(k)] = topk_metrics["success"]
            if k == PRIMARY_RECALL_K:
                print(f"    precision@{k} = {topk_metrics['precision']:.4f}")
                print(f"    ndcg@{k} = {topk_metrics['ndcg']:.4f}")
                print(f"    success@{k} = {topk_metrics['success']:.4f}")

        print("  Measuring filtered search duration …")
        f_lats = measure_durations(
            lambda q, k: query_arango(db, q, k, category_filter="cat_0"),
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
            "topk_metrics": run_topk_metrics,
            "fp50": fp50, "fp95": fp95,
        })
        save_fn()

    return BenchmarkResult(
        db_name="ArangoDB",
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
        precision_at_k={k: statistics.mean(v) for k, v in all_precision.items()},
        precision_at_k_std={k: (statistics.stdev(v) if len(v) > 1 else 0.0) for k, v in all_precision.items()},
        ndcg_at_k={k: statistics.mean(v) for k, v in all_ndcg.items()},
        ndcg_at_k_std={k: (statistics.stdev(v) if len(v) > 1 else 0.0) for k, v in all_ndcg.items()},
        success_at_k={k: statistics.mean(v) for k, v in all_success.items()},
        success_at_k_std={k: (statistics.stdev(v) if len(v) > 1 else 0.0) for k, v in all_success.items()},
        filtered_duration_p50_ms=statistics.mean(all_fp50),
        filtered_duration_p50_std=statistics.stdev(all_fp50) if len(all_fp50) > 1 else 0.0,
        filtered_duration_p95_ms=statistics.mean(all_fp95),
        filtered_duration_p95_std=statistics.stdev(all_fp95) if len(all_fp95) > 1 else 0.0,
        memory_mb=mem_mb,
        num_runs=num_runs,
    )
