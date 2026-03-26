#!/usr/bin/env bash
set -euo pipefail

echo "=== ArangoDB vs Qdrant Benchmark ==="

# 1. Start containers
echo "[1/4] Starting containers …"
docker compose up -d --wait
echo "      Containers ready."

# 2. Install Python deps
echo "[2/4] Installing Python dependencies …"
pip install -q -r requirements.txt

# 3. Wait for services to be healthy
echo "[3/4] Waiting for services …"
until curl -sf -u root:benchmark http://localhost:8529/_api/version > /dev/null 2>&1; do sleep 1; done
echo "      ArangoDB ready."
until curl -sf http://localhost:6333/healthz > /dev/null 2>&1; do sleep 1; done
echo "      Qdrant ready."

# 4. Run benchmark
echo "[4/4] Running benchmark …"
python main.py "$@"

echo ""
echo "Done! Charts are in ./results/"
