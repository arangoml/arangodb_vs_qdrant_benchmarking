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
PRIMARY_RECALL_K = 5
CONTAINER_MEMORY_GB = 5  # Memory limit per container (from docker-compose)
NUM_CATEGORIES = 10
CPU_COUNT = os.cpu_count() or 4
CONCURRENT_WORKERS = CPU_COUNT
WARMUP_QUERIES = 1
NUM_RUNS = 3  # Repeat measurements for statistical robustness
BATCH_SIZE = 500  # Sub-batch size for database inserts

HF_DATASET = "Cohere/beir-embed-english-v3"
HF_SUBSET = "nq"  # 2.68M passages, 3.4K queries, with qrels (~5.5 GB)

# Suppress noisy connection retry warnings
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
