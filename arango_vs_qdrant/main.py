"""
ArangoDB vs Qdrant — RAG Retrieval Benchmark
=============================================
Benchmarks: ingestion + index creation time, query duration (p50/p95/p99),
throughput (QPS), recall@k (vs human qrels), and filtered search.

Runs across multiple document counts and Qdrant HNSW configurations.
All queries and their relevant document IDs are used for comprehensive
accuracy testing at every dataset size.
"""

import json
import argparse
from pathlib import Path
from dataclasses import asdict

import numpy as np
from tabulate import tabulate

from config import NUM_RUNS, HF_SUBSET
from dataset import load_beir_dataset
from docker import start_container, stop_container
from measure import BenchmarkResult
from arango_bench import run_benchmark_arango
from qdrant_bench import run_benchmark_qdrant
from plotting import plot_results
from checkpoint import load_checkpoint, save_checkpoint


DOC_COUNTS = [1_000_000]

QDRANT_CONFIGS = [
    None,  # Default (Qdrant built-in: m=16, ef_construct=100, ef_search=128)
    {"m": 32, "ef_construct": 200, "ef_search": 128, "name": "Qdrant (m=32, efc=200, efs=128)"},
    {"m": 8,  "ef_construct": 50,  "ef_search": 32,  "name": "Qdrant (m=8, efc=50, efs=32)"},
    {"m": 24, "ef_construct": 150, "ef_search": 96,  "name": "Qdrant (m=24, efc=150, efs=96)"},
]


def _qdrant_ckpt_key(cfg: dict | None) -> str:
    if cfg is None:
        return "qdrant"
    return "qdrant_" + cfg["name"].lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("=", "")


def _is_complete(ckpt: dict, db_key: str, num_runs: int) -> bool:
    db = ckpt.get(db_key, {})
    if not db.get("ingest"):
        return False
    if len(db.get("runs", [])) < num_runs:
        return False
    return True


def _format_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    return f"{n // 1_000}K"


def _fmt(mean: float, std: float, fmt: str = ".1f") -> str:
    if std > 0.005:
        return f"{mean:{fmt}}\u00b1{std:{fmt}}"
    return f"{mean:{fmt}}"


def print_summary(all_results: list[BenchmarkResult], num_runs: int):
    print(f"\n  Values are mean\u00b1stddev across {num_runs} runs.\n")
    rows = []
    for r in all_results:
        total_load_s = r.ingest_time_s + r.index_build_time_s
        rows.append([
            r.db_name,
            f"{r.dataset_size:,}",
            f"{total_load_s:.2f}",
            _fmt(r.duration_p50_ms, r.duration_p50_std),
            _fmt(r.duration_p95_ms, r.duration_p95_std),
            _fmt(r.duration_p99_ms, r.duration_p99_std),
            _fmt(r.throughput_qps, r.throughput_qps_std),
            _fmt(r.recall_at_k.get(10, 0), r.recall_at_k_std.get(10, 0), fmt=".4f"),
            _fmt(r.filtered_duration_p50_ms, r.filtered_duration_p50_std),
            f"{r.memory_mb:.1f}",
        ])
    headers = ["DB", "N", "Ingest+Idx(s)", "p50(ms)", "p95(ms)", "p99(ms)",
               "QPS", "R@10", "Filt p50(ms)", "Mem(MB)"]
    print(tabulate(rows, headers=headers, tablefmt="github"))


def _rebuild_from_checkpoint(db_ckpt: dict, db_name: str, num_runs: int) -> BenchmarkResult:
    import statistics as stats
    from config import TOP_K_VALUES

    ingest = db_ckpt["ingest"]
    runs = db_ckpt["runs"]

    all_p50 = [r["p50"] for r in runs]
    all_p95 = [r["p95"] for r in runs]
    all_p99 = [r["p99"] for r in runs]
    all_qps = [r["qps"] for r in runs]
    all_fp50 = [r["fp50"] for r in runs]
    all_fp95 = [r["fp95"] for r in runs]
    all_recall = {k: [] for k in TOP_K_VALUES}
    for r in runs:
        for k in TOP_K_VALUES:
            all_recall[k].append(r["recall"][str(k)])

    return BenchmarkResult(
        db_name=db_name,
        dataset_size=ingest["dataset_size"],
        ingest_time_s=ingest["ingest_time_s"],
        index_build_time_s=ingest["index_build_time_s"],
        duration_p50_ms=stats.mean(all_p50),
        duration_p50_std=stats.stdev(all_p50) if len(all_p50) > 1 else 0.0,
        duration_p95_ms=stats.mean(all_p95),
        duration_p95_std=stats.stdev(all_p95) if len(all_p95) > 1 else 0.0,
        duration_p99_ms=stats.mean(all_p99),
        duration_p99_std=stats.stdev(all_p99) if len(all_p99) > 1 else 0.0,
        throughput_qps=stats.mean(all_qps),
        throughput_qps_std=stats.stdev(all_qps) if len(all_qps) > 1 else 0.0,
        recall_at_k={k: stats.mean(v) for k, v in all_recall.items()},
        recall_at_k_std={k: (stats.stdev(v) if len(v) > 1 else 0.0) for k, v in all_recall.items()},
        filtered_duration_p50_ms=stats.mean(all_fp50),
        filtered_duration_p50_std=stats.stdev(all_fp50) if len(all_fp50) > 1 else 0.0,
        filtered_duration_p95_ms=stats.mean(all_fp95),
        filtered_duration_p95_std=stats.stdev(all_fp95) if len(all_fp95) > 1 else 0.0,
        memory_mb=ingest["memory_mb"],
        num_runs=num_runs,
    )


def run_for_doc_count(doc_count: int, args) -> list[BenchmarkResult]:
    label = _format_count(doc_count)
    out_dir = Path(args.out) / label
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        import shutil
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {}
    else:
        ckpt = load_checkpoint(out_dir)
        if ckpt:
            print(f"  Found checkpoint for {label} — will resume.")

    arango_done = _is_complete(ckpt, "arango", args.runs)
    qdrant_status = {
        _qdrant_ckpt_key(cfg): _is_complete(ckpt, _qdrant_ckpt_key(cfg), args.runs)
        for cfg in QDRANT_CONFIGS
    }

    need_arango = args.only != "qdrant" and not arango_done
    need_qdrant = args.only != "arango" and not all(qdrant_status.values())

    corpus = queries = qrels = None
    all_query_vecs = all_query_ids = None

    if need_arango or need_qdrant:
        print(f"\n  Loading BEIR dataset: {args.subset} ({doc_count:,} docs)")
        corpus, queries, qrels = load_beir_dataset(args.subset, target_doc_count=doc_count)
        usable_qids = [qid for qid in qrels if qid in queries]
        all_query_vecs = np.array(
            [queries[qid]["embedding"] for qid in usable_qids], dtype=np.float32,
        )
        all_query_ids = usable_qids
        print(f"  Query pool: {len(usable_qids)} queries (all used per run)")

    all_results: list[BenchmarkResult] = []

    def make_save_fn():
        def _save():
            save_checkpoint(out_dir, ckpt)
        return _save

    # ── ArangoDB ──
    if args.only != "qdrant":
        if arango_done:
            print(f"  ArangoDB ({label}) already complete.")
            result = _rebuild_from_checkpoint(ckpt["arango"], "ArangoDB", args.runs)
        else:
            ckpt.setdefault("arango", {})
            has_ingest = bool(ckpt["arango"].get("ingest"))
            start_container("arangodb", preserve_volumes=has_ingest)
            try:
                result = run_benchmark_arango(
                    corpus, all_query_vecs, all_query_ids, qrels,
                    num_runs=args.runs, ckpt_db=ckpt["arango"], save_fn=make_save_fn(),
                )
            except Exception as e:
                import subprocess
                print(f"\n  !!! ArangoDB benchmark failed: {e}")
                try:
                    info = subprocess.run(
                        ["docker", "inspect", "arango-bench",
                         "--format",
                         "OOMKilled={{.State.OOMKilled}} "
                         "ExitCode={{.State.ExitCode}} "
                         "Status={{.State.Status}} "
                         "RestartCount={{.RestartCount}}"],
                        capture_output=True, text=True, timeout=5,
                    )
                    print(f"  Container state: {info.stdout.strip()}")
                except Exception:
                    print("  (could not inspect container)")
                try:
                    logs = subprocess.run(
                        ["docker", "logs", "--tail", "50", "arango-bench"],
                        capture_output=True, text=True, timeout=10,
                    )
                    print(f"  Last 50 lines of container logs:\n{logs.stdout}{logs.stderr}")
                except Exception:
                    print("  (could not fetch container logs)")
                raise
            finally:
                stop_container("arangodb")
        all_results.append(result)

    # ── Qdrant (all configs) ──
    if args.only != "arango":
        any_needs_work = not all(qdrant_status.values())
        if any_needs_work:
            has_any_ingest = any(
                ckpt.get(_qdrant_ckpt_key(cfg), {}).get("ingest")
                for cfg in QDRANT_CONFIGS
            )
            start_container("qdrant", preserve_volumes=has_any_ingest)

        try:
            for cfg in QDRANT_CONFIGS:
                key = _qdrant_ckpt_key(cfg)
                db_label = cfg["name"] if cfg else "Qdrant (Default)"

                if qdrant_status[key]:
                    print(f"  {db_label} ({label}) already complete.")
                    result = _rebuild_from_checkpoint(ckpt[key], db_label, args.runs)
                else:
                    ckpt.setdefault(key, {})
                    result = run_benchmark_qdrant(
                        corpus, all_query_vecs, all_query_ids, qrels,
                        num_runs=args.runs, ckpt_db=ckpt[key], save_fn=make_save_fn(),
                        hnsw_config=cfg,
                    )
                all_results.append(result)
        except Exception as e:
            import subprocess
            print(f"\n  !!! Qdrant benchmark failed: {e}")
            try:
                info = subprocess.run(
                    ["docker", "inspect", "qdrant-bench",
                     "--format",
                     "OOMKilled={{.State.OOMKilled}} "
                     "ExitCode={{.State.ExitCode}} "
                     "Status={{.State.Status}} "
                     "RestartCount={{.RestartCount}}"],
                    capture_output=True, text=True, timeout=5,
                )
                print(f"  Container state: {info.stdout.strip()}")
            except Exception:
                print("  (could not inspect container)")
            try:
                logs = subprocess.run(
                    ["docker", "logs", "--tail", "30", "qdrant-bench"],
                    capture_output=True, text=True, timeout=10,
                )
                print(f"  Last 30 lines of container logs:\n{logs.stdout}{logs.stderr}")
            except Exception:
                print("  (could not fetch container logs)")
            raise
        finally:
            if any_needs_work:
                stop_container("qdrant")

    # ── Save & Plot ──
    if all_results:
        print_summary(all_results, args.runs)

    raw = [asdict(r) for r in all_results]
    with open(out_dir / "results.json", "w") as f:
        json.dump({"benchmarks": raw}, f, indent=2)
    print(f"\n  Results saved to {out_dir}/results.json")

    if len(all_results) >= 2:
        plot_results(all_results, out_dir)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="ArangoDB vs Qdrant RAG Retrieval Benchmark",
    )
    parser.add_argument("--out", type=str, default="results",
                        help="Base output directory")
    parser.add_argument("--only", choices=["arango", "qdrant"], default=None,
                        help="Run only one database type")
    parser.add_argument("--subset", type=str, default=HF_SUBSET,
                        help="BEIR subset (default: nq)")
    parser.add_argument("--runs", type=int, default=NUM_RUNS,
                        help="Measurement runs per config (default: 3)")
    parser.add_argument("--fresh", action="store_true",
                        help="Discard previous results and start fresh")
    parser.add_argument("--doc-counts", type=int, nargs="+", default=None,
                        help=f"Doc counts to test (default: {DOC_COUNTS})")
    args = parser.parse_args()

    doc_counts = args.doc_counts or DOC_COUNTS

    print(f"\n{'#'*60}")
    print(f"  BENCHMARK")
    print(f"  Doc counts : {[_format_count(n) for n in doc_counts]}")
    print(f"  Qdrant cfgs: {[cfg['name'] if cfg else 'Default' for cfg in QDRANT_CONFIGS]}")
    print(f"  Runs/config: {args.runs}")
    print(f"{'#'*60}")

    grand_results = {}
    for doc_count in doc_counts:
        print(f"\n{'#'*60}")
        print(f"  === {_format_count(doc_count)} DOCUMENTS ===")
        print(f"{'#'*60}")
        results = run_for_doc_count(doc_count, args)
        grand_results[doc_count] = results

    # Combined cross-doc-count results
    if len(doc_counts) > 1:
        base_out = Path(args.out)
        all_flat = []
        for dc in doc_counts:
            all_flat.extend(grand_results.get(dc, []))
        if all_flat:
            raw = [asdict(r) for r in all_flat]
            with open(base_out / "all_results.json", "w") as f:
                json.dump({"benchmarks": raw}, f, indent=2)
            print(f"\n  Combined results saved to {base_out}/all_results.json")

    print("\n  All done!\n")


if __name__ == "__main__":
    main()
