"""
Benchmark configuration constants.
"""

import os
import logging

ARANGO_HOST = "http://localhost:8529"
ARANGO_DB = "benchmark"
ARANGO_USER = "root"
ARANGO_PASS = "benchmark"
ARANGO_COLLECTION = "documents"

QDRANT_HOST = "localhost"
QDRANT_HTTP_PORT = 6333
QDRANT_GRPC_PORT = 6334

DIMENSIONS = 1024  # Cohere embed-english-v3.0
TOP_K_VALUES = [1, 5, 10, 20, 50]
CONTAINER_MEMORY_GB = 4  # Memory limit per container (from docker-compose)
TARGET_DOC_COUNT = 1_000_000  # Total documents to insert (real + synthetic)
FILL_BATCH_SIZE = 500_000  # Documents per batch during the fill loop
NUM_QUERIES = 200
NUM_CATEGORIES = 10
CPU_COUNT = os.cpu_count() or 4
CONCURRENT_WORKERS = CPU_COUNT
WARMUP_QUERIES = 50
NUM_RUNS = 3  # Repeat measurements for statistical robustness
HF_DATASET = "Cohere/beir-embed-english-v3"
HF_SUBSET = "fiqa"  # 57.6K docs, 6.6K queries, with qrels

# Sweep parameters for recall-vs-duration Pareto curves
NPROBE_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128, 256]  # capped at nLists at runtime
EF_SWEEP = [16, 32, 50, 64, 100, 128, 200, 256]

# Suppress noisy connection retry warnings
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
