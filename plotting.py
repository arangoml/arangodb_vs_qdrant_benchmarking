"""
Chart generation for benchmark results.
"""

from pathlib import Path

import numpy as np

from measure import BenchmarkResult, RecallDurationPoint


def plot_results(all_results: list[BenchmarkResult], out_dir: Path,
                 pareto_points: list[RecallDurationPoint] | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=1.1)
    out_dir.mkdir(parents=True, exist_ok=True)

    a = next((r for r in all_results if r.db_name == "ArangoDB"), None)
    q = next((r for r in all_results if r.db_name == "Qdrant"), None)
    if not a or not q:
        print("  Skipping plots — need results from both databases.")
        return

    n = a.dataset_size
    colors = {"ArangoDB": "#68A063", "Qdrant": "#DC382C"}
    label_a = "ArangoDB (IVF, HTTP)"
    label_q = "Qdrant (HNSW, gRPC)"

    # Compute the hyperparameters that were used (same formulas as arango.py)
    n_lists = max(1, int(n ** 0.5))
    n_probe = max(1, n_lists // 4)
    footnote = (
        f"ArangoDB: IVF index (nLists={n_lists}, defaultNProbe={n_probe}, trainingIter=25), HTTP/REST.  "
        f"Qdrant: HNSW index (default m=16, ef_construct=100), gRPC.  "
        f"Each DB uses its best available protocol."
    )

    # ---- 1. Indexing Time — bar chart ----
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar([label_a, label_q], [a.index_time_s, q.index_time_s],
                  color=[colors["ArangoDB"], colors["Qdrant"]], width=0.5)
    ax.bar_label(bars, fmt="%.1fs", padding=3)
    ax.set_ylabel("Indexing Time (s)")
    ax.set_title(f"Indexing Time  (N={n:,})")
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=7, style="italic", color="grey")
    fig.tight_layout()
    fig.savefig(out_dir / "01_indexing_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 2. Query Duration (p50, p95, p99) — grouped bar ----
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3)
    w = 0.35
    bars_a = ax.bar(x - w / 2, [a.duration_p50_ms, a.duration_p95_ms, a.duration_p99_ms],
                    w, label=label_a, color=colors["ArangoDB"])
    bars_q = ax.bar(x + w / 2, [q.duration_p50_ms, q.duration_p95_ms, q.duration_p99_ms],
                    w, label=label_q, color=colors["Qdrant"])
    ax.bar_label(bars_a, fmt="%.1f", padding=3, fontsize=8)
    ax.bar_label(bars_q, fmt="%.1f", padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["p50", "p95", "p99"])
    ax.set_ylabel("Duration (ms)")
    ax.set_title(f"Query Duration  (N={n:,})")
    ax.legend()
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=7, style="italic", color="grey")
    fig.tight_layout()
    fig.savefig(out_dir / "02_query_duration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 3. Throughput (QPS) — bar chart ----
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar([label_a, label_q], [a.throughput_qps, q.throughput_qps],
                  color=[colors["ArangoDB"], colors["Qdrant"]], width=0.5)
    ax.bar_label(bars, fmt="%.0f", padding=3)
    ax.set_ylabel("Queries Per Second")
    ax.set_title(f"Throughput (QPS)  (N={n:,})")
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=7, style="italic", color="grey")
    fig.tight_layout()
    fig.savefig(out_dir / "03_throughput.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 4. Recall@K ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = sorted(a.recall_at_k.keys())
    ax.plot(ks, [a.recall_at_k[k] for k in ks],
            "o-", label=label_a, color=colors["ArangoDB"], linewidth=2, markersize=8)
    ax.plot(ks, [q.recall_at_k[k] for k in ks],
            "s-", label=label_q, color=colors["Qdrant"], linewidth=2, markersize=8)
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K")
    ax.set_title(f"Recall@K — BEIR/FIQA  (N={n:,})")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=7, style="italic", color="grey")
    fig.tight_layout()
    fig.savefig(out_dir / "04_recall_at_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 5. Filtered Search Duration — bar chart ----
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    w = 0.35
    bars_a = ax.bar(x - w / 2, [a.filtered_duration_p50_ms, a.filtered_duration_p95_ms],
                    w, label=label_a, color=colors["ArangoDB"])
    bars_q = ax.bar(x + w / 2, [q.filtered_duration_p50_ms, q.filtered_duration_p95_ms],
                    w, label=label_q, color=colors["Qdrant"])
    ax.bar_label(bars_a, fmt="%.1f", padding=3, fontsize=8)
    ax.bar_label(bars_q, fmt="%.1f", padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["p50", "p95"])
    ax.set_ylabel("Duration (ms)")
    ax.set_title(f"Filtered Search Duration  (N={n:,})")
    ax.legend()
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=7, style="italic", color="grey")
    fig.tight_layout()
    fig.savefig(out_dir / "05_filtered_duration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 6. Summary Radar / Spider chart ----
    metrics_labels = ["Index Speed", "Duration (p50)", "Duration (p99)",
                      "Throughput", "Recall@10", "Filtered Lat."]

    def norm_inv(val_a, val_b):
        """Lower is better — invert so higher = better."""
        mn = min(val_a, val_b) or 1e-9
        return mn / (val_a or 1e-9), mn / (val_b or 1e-9)

    def norm_dir(val_a, val_b):
        """Higher is better."""
        mx = max(val_a, val_b, 1e-9)
        return val_a / mx, val_b / mx

    a_vals, q_vals = [], []
    va, vq = norm_inv(a.index_time_s, q.index_time_s)
    a_vals.append(va); q_vals.append(vq)
    va, vq = norm_inv(a.duration_p50_ms, q.duration_p50_ms)
    a_vals.append(va); q_vals.append(vq)
    va, vq = norm_inv(a.duration_p99_ms, q.duration_p99_ms)
    a_vals.append(va); q_vals.append(vq)
    va, vq = norm_dir(a.throughput_qps, q.throughput_qps)
    a_vals.append(va); q_vals.append(vq)
    va, vq = norm_dir(a.recall_at_k.get(10, 0), q.recall_at_k.get(10, 0))
    a_vals.append(va); q_vals.append(vq)
    va, vq = norm_inv(a.filtered_duration_p50_ms, q.filtered_duration_p50_ms)
    a_vals.append(va); q_vals.append(vq)

    angles = np.linspace(0, 2 * np.pi, len(metrics_labels), endpoint=False).tolist()
    a_vals += a_vals[:1]
    q_vals += q_vals[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, a_vals, "o-", label=label_a, color=colors["ArangoDB"], linewidth=2)
    ax.fill(angles, a_vals, alpha=0.15, color=colors["ArangoDB"])
    ax.plot(angles, q_vals, "s-", label=label_q, color=colors["Qdrant"], linewidth=2)
    ax.fill(angles, q_vals, alpha=0.15, color=colors["Qdrant"])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_labels)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Overall Comparison — BEIR/FIQA  (N={n:,})", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=7, style="italic", color="grey")
    fig.tight_layout()
    fig.savefig(out_dir / "06_radar_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 7. Memory Usage — bar chart ----
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar([label_a, label_q], [a.memory_mb, q.memory_mb],
                  color=[colors["ArangoDB"], colors["Qdrant"]], width=0.5)
    ax.bar_label(bars, fmt="%.0f MB", padding=3)
    ax.set_ylabel("Memory Usage (MB)")
    ax.set_title(f"Memory Usage  (N={n:,})")
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=7, style="italic", color="grey")
    fig.tight_layout()
    fig.savefig(out_dir / "07_memory_usage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 8. Recall vs Duration Pareto Curves ----
    if pareto_points:
        a_pts = sorted(
            [p for p in pareto_points if p.db_name == "ArangoDB"],
            key=lambda p: p.recall_at_10,
        )
        q_pts = sorted(
            [p for p in pareto_points if p.db_name == "Qdrant"],
            key=lambda p: p.recall_at_10,
        )
        if a_pts or q_pts:
            fig, ax = plt.subplots(figsize=(9, 6))
            if a_pts:
                ax.plot(
                    [p.recall_at_10 for p in a_pts],
                    [p.duration_p50_ms for p in a_pts],
                    "o-", label="ArangoDB (IVF, nProbe sweep)",
                    color=colors["ArangoDB"], linewidth=2, markersize=7,
                )
                for p in a_pts:
                    ax.annotate(
                        f"nP={p.param_value}", (p.recall_at_10, p.duration_p50_ms),
                        textcoords="offset points", xytext=(5, 5), fontsize=7, color=colors["ArangoDB"],
                    )
            if q_pts:
                ax.plot(
                    [p.recall_at_10 for p in q_pts],
                    [p.duration_p50_ms for p in q_pts],
                    "s-", label="Qdrant (HNSW, ef sweep)",
                    color=colors["Qdrant"], linewidth=2, markersize=7,
                )
                for p in q_pts:
                    ax.annotate(
                        f"ef={p.param_value}", (p.recall_at_10, p.duration_p50_ms),
                        textcoords="offset points", xytext=(5, 5), fontsize=7, color=colors["Qdrant"],
                    )
            ax.set_xlabel("Recall@10")
            ax.set_ylabel("Query Duration p50 (ms)")
            ax.set_title(f"Recall vs Duration Tradeoff — N={n:,}")
            ax.legend()
            ax.set_xlim(0, 1.05)
            fig.text(
                0.5, -0.02,
                "Lower-right is better (high recall, low duration). Each point is a different nProbe / ef setting.",
                ha="center", fontsize=7, style="italic", color="grey",
            )
            fig.tight_layout()
            fig.savefig(out_dir / "08_pareto_recall_duration.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    print(f"\n  Charts saved to {out_dir}/")
