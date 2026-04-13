@echo off
setlocal

echo === ArangoDB vs Qdrant Benchmark ===

REM 1. Start containers
echo [1/4] Starting containers ...
docker compose up -d --wait
echo       Containers ready.

REM 2. Install Python deps
echo [2/4] Installing Python dependencies ...
pip install -q -r requirements.txt

REM 3. Wait for services to be healthy
echo [3/4] Waiting for services ...

:wait_arango
curl -sf -u root:benchmark http://localhost:8529/_api/version >nul 2>&1
if errorlevel 1 (
    timeout /t 1 /nobreak >nul
    goto wait_arango
)
echo       ArangoDB ready.

:wait_qdrant
curl -sf http://localhost:6333/healthz >nul 2>&1
if errorlevel 1 (
    timeout /t 1 /nobreak >nul
    goto wait_qdrant
)
echo       Qdrant ready.

REM 4. Run benchmark
echo [4/4] Running benchmark ...
python main.py %*

echo.
echo Done! Charts are in ./results/
