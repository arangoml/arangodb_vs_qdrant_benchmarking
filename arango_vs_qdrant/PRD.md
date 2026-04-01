# ArangoDB vs Qdrant — RAG Retrieval Benchmark

## Objective

Compare ArangoDB and Qdrant as vector stores for RAG retrieval pipelines. The benchmark measures ingestion speed, query latency, throughput, recall accuracy, filtered search performance, and memory usage under identical conditions.

## Dataset

- **Source:** [Cohere/beir-embed-english-v3](https://huggingface.co/datasets/Cohere/beir-embed-english-v3) (NQ subset)
- **Embeddings:** Cohere embed-english-v3.0, 1024 dimensions
- **Scale tested:** 100K, 200K, 1M documents
- **Queries:** All usable queries from BEIR qrels (~3.4K for NQ)
- **Ground truth:** Human relevance judgments (qrels) from BEIR — not synthetic nearest-neighbor ground truth

## Infrastructure

Both databases run as Docker containers with identical resource constraints:

| Setting | Value |
|---|---|
| Memory limit | 5 GB per container |
| ArangoDB version | 3.12 |
| Qdrant version | 1.17.0 |
| ArangoDB protocol | HTTP |
| Qdrant protocol | gRPC |

## Index Parameters

### ArangoDB (IVF)

| Parameter | Value |
|---|---|
| Index type | `vector` (IVF) |
| Metric | Cosine |
| nLists | `15 * sqrt(N)` (dynamic, based on dataset size) |
| defaultNProbe | `sqrt(nLists)` |
| Training iterations | 25 |

### Qdrant (HNSW)

Four configurations tested:

| Config | m | ef_construct | ef_search |
|---|---|---|---|
| Default | 16 | 100 | 128 |
| High accuracy | 32 | 200 | 128 |
| Low resource | 8 | 50 | 32 |
| Balanced | 24 | 150 | 96 |

## What Is Measured

Each configuration is measured across 3 independent runs (mean ± stddev reported):

| Metric | Description |
|---|---|
| Ingestion + index build time | Wall-clock time to insert all documents and build the vector index |
| Query latency (p50, p95, p99) | Sequential single-query latency in ms, with 50-query warmup |
| Throughput (QPS) | Queries per second under concurrent load (N workers = CPU count) |
| Recall@k | Fraction of human-judged relevant documents in top-k results (k = 1, 5, 10, 20, 50) |
| Filtered search latency | Query latency with a keyword payload filter (`category`) |
| Memory usage | Container RSS via Docker stats, measured after ingestion + indexing |

## Methodology

1. Load dataset from HuggingFace (parquet shards via DuckDB)
2. Start database container (Docker Compose)
3. Ingest corpus in batches of 500 documents
4. Build vector index and wait for completion
5. Record memory usage
6. Run 3 measurement passes, each consisting of:
   - Sequential latency measurement (all queries, with warmup)
   - Concurrent throughput measurement
   - Recall@k computation against BEIR qrels
   - Filtered search latency measurement
7. Aggregate results (mean ± stddev) and generate comparison plots

Checkpointing is supported — interrupted runs resume from the last completed step.

## Output

- `results.json` — raw benchmark data per doc-count
- 7 comparison plots (load time, query duration, throughput, recall@k, filtered duration, radar summary, memory usage)
