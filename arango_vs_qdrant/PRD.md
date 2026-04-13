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

Methodology notes from official vendor docs used by this harness:

- Qdrant filtered-search setup creates the `category` payload index before uploading any data, following Qdrant's filterable HNSW guidance: <https://qdrant.tech/course/essentials/day-2/filterable-hnsw/>
- Qdrant's official benchmark harness does not use a separate warmup-query phase before timing search requests; it initializes the client and then measures all queries directly: <https://github.com/qdrant/vector-db-benchmark/blob/master/engine/base_client/search.py> and <https://github.com/qdrant/vector-db-benchmark/blob/master/engine/clients/qdrant/search.py>
- ArangoDB vector index tuning intentionally uses a `defaultNProbe` higher than the built-in default because Arango's vector index docs state the default is `1` and that you should generally use a higher value or pass `nProbe` per query: <https://docs.arango.ai/arangodb/3.12/indexes-and-search/indexing/working-with-indexes/vector-indexes/>

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
| Query latency (p50, p95, p99) | Sequential single-query latency in ms, without a separate warmup phase |
| Throughput (QPS) | Queries per second under concurrent load (N workers = CPU count) |
| Recall@k | Fraction of human-judged relevant documents in top-k results; `recall@5` is the primary reported recall metric |
| Precision@k | Fraction of the top-k results that are human-judged relevant; `precision@5` is the primary reported precision metric |
| nDCG@k | Ranking quality at top-k with higher weight on earlier relevant hits; `nDCG@5` is the primary reported nDCG metric |
| Success@k | Fraction of queries with at least one relevant document in the top-k results; `success@5` is the primary reported success metric |
| Filtered search latency | Query latency with a keyword payload filter (`category`) |
| Memory usage | Container RSS via Docker stats, measured after ingestion + indexing |

## Methodology

1. Load dataset from HuggingFace (parquet shards via DuckDB)
2. Start database container (Docker Compose)
3. Ingest corpus in batches of 500 documents
4. Build vector index and wait for completion
5. Record memory usage
6. Run 3 measurement passes, each consisting of:
   - Sequential latency measurement (all queries, without a separate warmup phase)
   - Concurrent throughput measurement
   - Recall@k plus low-k quality metrics (`precision@5`, `nDCG@5`, `success@5`) computed from the same retrieved result sets against BEIR qrels
   - Filtered search latency measurement
7. Aggregate results (mean ± stddev) and generate comparison plots

Implementation details for fairness:

- Search quality is evaluated by identifiers. ArangoDB returns `doc_id` from AQL; Qdrant point IDs are mapped back to corpus document IDs client-side.
- Only the summed load metric (`ingestion + index build`) is treated as directly comparable across engines. The internal split between ingest time and index completion time is engine-specific because Qdrant performs much of HNSW construction during synchronous upserts, while ArangoDB creates its IVF index after document insertion.
- Cross-database comparisons are intended within a given document count. If multiple document counts are tested, the usable query set may differ by corpus size because BEIR qrels are kept only for queries whose judged-relevant documents are present in that corpus.

Checkpointing is supported — interrupted runs resume from the last completed step.

## Output

- `results.json` — raw benchmark data per doc-count
- 10 comparison plots (load time, query duration, throughput, recall@k, filtered duration, radar summary, memory usage, precision@k, nDCG@k, success@k)
