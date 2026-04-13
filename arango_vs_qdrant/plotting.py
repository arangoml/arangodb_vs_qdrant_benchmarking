"""
Chart generation for benchmark results.
"""

from pathlib import Path

import numpy as np

from config import CONTAINER_MEMORY_GB, PRIMARY_RECALL_K
from measure import BenchmarkResult


COLORS = ["#68A063", "#DC382C", "#1E88E5", "#FFC107", "#7B1FA2"]
MARKERS = ["o", "s", "^", "D", "v"]


def _plot_metric_at_k(fig, ax, all_results, labels, colors, n, footnote, attr_name, ylabel, title):
    for i, r in enumerate(all_results):
        values = getattr(r, attr_name)
        ks = sorted(values.keys())
        ax.plot(
            ks,
            [values[k] for k in ks],
            f"{MARKERS[i % len(MARKERS)]}-",
            label=labels[i],
            color=colors[i],
            linewidth=2,
            markersize=7,
        )
    ax.set_xlabel("K")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  (N={n:,})")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=7, style="italic", color="grey")
    fig.tight_layout()


def plot_results(all_results: list[BenchmarkResult], out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=1.1)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(all_results) < 2:
        print("  Skipping plots — need at least 2 results.")
        return

    n = all_results[0].dataset_size
    labels = [r.db_name for r in all_results]
    colors = [COLORS[i % len(COLORS)] for i in range(len(all_results))]

    footnote = f"RAM: {CONTAINER_MEMORY_GB} GB per container. Each DB uses its best available protocol."

    # ---- 1. Total Load Time ----
    fig, ax = plt.subplots(figsize=(max(7, len(all_results) * 1.5), 5))
    vals = [r.ingest_time_s + r.index_build_time_s for r in all_results]
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    ax.bar_label(bars, fmt="%.1fs", padding=3, fontsize=8)
    ax.set_ylabel("Time (s)")
    ax.set_title(f"Ingestion + Index Creation  (N={n:,})")
    plt.xticks(rotation=25, ha="right")
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=7, style="italic", color="grey")
    fig.tight_layout()
    fig.savefig(out_dir / "01_load_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 2. Query Duration (p50, p95, p99) ----
    fig, ax = plt.subplots(figsize=(max(9, len(all_results) * 2), 5))
    x = np.arange(3)
    w = 0.8 / len(all_results)
    for i, r in enumerate(all_results):
        offset = (i - len(all_results) / 2 + 0.5) * w
        b = ax.bar(x + offset,
                   [r.duration_p50_ms, r.duration_p95_ms, r.duration_p99_ms],
                   w, label=labels[i], color=colors[i])
        ax.bar_label(b, fmt="%.1f", padding=2, fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(["p50", "p95", "p99"])
    ax.set_ylabel("Duration (ms)")
    ax.set_title(f"Query Duration  (N={n:,})")
    ax.legend(fontsize=8)
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=7, style="italic", color="grey")
    fig.tight_layout()
    fig.savefig(out_dir / "02_query_duration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 3. Throughput (QPS) ----
    fig, ax = plt.subplots(figsize=(max(7, len(all_results) * 1.5), 5))
    vals = [r.throughput_qps for r in all_results]
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
    ax.set_ylabel("Queries Per Second")
    ax.set_title(f"Throughput (QPS)  (N={n:,})")
    plt.xticks(rotation=25, ha="right")
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=7, style="italic", color="grey")
    fig.tight_layout()
    fig.savefig(out_dir / "03_throughput.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 4. Recall@K ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, r in enumerate(all_results):
        ks = sorted(r.recall_at_k.keys())
        ax.plot(ks, [r.recall_at_k[k] for k in ks],
                f"{MARKERS[i % len(MARKERS)]}-", label=labels[i],
                color=colors[i], linewidth=2, markersize=7)
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K")
    ax.set_title(f"Recall@K  (N={n:,})")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=7, style="italic", color="grey")
    fig.tight_layout()
    fig.savefig(out_dir / "04_recall_at_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 5. Filtered Search Duration ----
    fig, ax = plt.subplots(figsize=(max(9, len(all_results) * 2), 5))
    x = np.arange(2)
    w = 0.8 / len(all_results)
    for i, r in enumerate(all_results):
        offset = (i - len(all_results) / 2 + 0.5) * w
        b = ax.bar(x + offset,
                   [r.filtered_duration_p50_ms, r.filtered_duration_p95_ms],
                   w, label=labels[i], color=colors[i])
        ax.bar_label(b, fmt="%.1f", padding=2, fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(["p50", "p95"])
    ax.set_ylabel("Duration (ms)")
    ax.set_title(f"Filtered Search Duration  (N={n:,})")
    ax.legend(fontsize=8)
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=7, style="italic", color="grey")
    fig.tight_layout()
    fig.savefig(out_dir / "05_filtered_duration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 6. Radar / Spider chart ----
    metrics_labels = ["Ingest+Idx", "p50 Lat", "p99 Lat",
                      "Throughput", f"Recall@{PRIMARY_RECALL_K}", "Filt. Lat"]
    lower_better = {0, 1, 2, 5}

    raw = []
    for r in all_results:
        raw.append([
            r.ingest_time_s + r.index_build_time_s,
            r.duration_p50_ms,
            r.duration_p99_ms,
            r.throughput_qps,
            r.recall_at_k.get(PRIMARY_RECALL_K, 0),
            r.filtered_duration_p50_ms,
        ])

    normed = [[] for _ in all_results]
    for mi in range(len(metrics_labels)):
        col = [raw[ri][mi] for ri in range(len(all_results))]
        if mi in lower_better:
            mn = min(col) or 1e-9
            for ri in range(len(all_results)):
                normed[ri].append(mn / (col[ri] or 1e-9))
        else:
            mx = max(col) or 1e-9
            for ri in range(len(all_results)):
                normed[ri].append(col[ri] / mx)

    angles = np.linspace(0, 2 * np.pi, len(metrics_labels), endpoint=False).tolist()
    for ri in range(len(all_results)):
        normed[ri].append(normed[ri][0])
    angles.append(angles[0])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, r in enumerate(all_results):
        ax.plot(angles, normed[i], f"{MARKERS[i % len(MARKERS)]}-",
                label=labels[i], color=colors[i], linewidth=2)
        ax.fill(angles, normed[i], alpha=0.06, color=colors[i])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_labels, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Overall Comparison  (N={n:,})", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=7, style="italic", color="grey")
    fig.tight_layout()
    fig.savefig(out_dir / "06_radar_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 7. Memory Usage ----
    fig, ax = plt.subplots(figsize=(max(7, len(all_results) * 1.5), 5))
    vals = [r.memory_mb for r in all_results]
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    ax.bar_label(bars, fmt="%.0f MB", padding=3, fontsize=8)
    ax.set_ylabel("Memory Usage (MB)")
    ax.set_title(f"Memory Usage  (N={n:,})")
    plt.xticks(rotation=25, ha="right")
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=7, style="italic", color="grey")
    fig.tight_layout()
    fig.savefig(out_dir / "07_memory_usage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 8. Precision@K ----
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_metric_at_k(
        fig, ax, all_results, labels, colors, n, footnote,
        "precision_at_k", "Precision@K", "Precision@K",
    )
    fig.savefig(out_dir / "08_precision_at_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 9. nDCG@K ----
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_metric_at_k(
        fig, ax, all_results, labels, colors, n, footnote,
        "ndcg_at_k", "nDCG@K", "nDCG@K",
    )
    fig.savefig(out_dir / "09_ndcg_at_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 10. Success@K ----
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_metric_at_k(
        fig, ax, all_results, labels, colors, n, footnote,
        "success_at_k", "Success@K", "Success@K",
    )
    fig.savefig(out_dir / "10_success_at_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  Charts saved to {out_dir}/")
