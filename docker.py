"""
Docker container lifecycle management.
"""

import time
from pathlib import Path

COMPOSE_DIR = Path(__file__).resolve().parent  # directory containing docker-compose.yml


def _run_compose(cmd: str, timeout: int = 120):
    """Run a docker compose command in the project directory."""
    import subprocess
    result = subprocess.run(
        f"docker compose {cmd}",
        shell=True, cwd=COMPOSE_DIR,
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker compose {cmd} failed:\n{result.stderr}")
    return result.stdout


def _free_ports(service: str):
    """Stop any Docker container already occupying the ports needed by *service*."""
    import subprocess
    port_map = {
        "arangodb": [8529],
        "qdrant": [6333, 6334],
    }
    for port in port_map.get(service, []):
        result = subprocess.run(
            ["docker", "ps", "--filter", f"publish={port}", "-q"],
            capture_output=True, text=True,
        )
        for cid in result.stdout.strip().splitlines():
            cid = cid.strip()
            if cid:
                print(f"  Port {port} in use by container {cid} — stopping it …")
                subprocess.run(["docker", "rm", "-f", cid],
                               capture_output=True, text=True)


def start_container(service: str, preserve_volumes: bool = False):
    """Start a single service container.

    If preserve_volumes is True, keep existing Docker volumes (for resume).
    Otherwise wipe volumes for a clean start.
    """
    print(f"\n  Starting {service} container (stopping others) …")
    try:
        if preserve_volumes:
            _run_compose("down", timeout=60)  # stop containers but keep volumes
        else:
            _run_compose("down -v", timeout=60)  # stop and remove volumes
    except Exception:
        pass  # may not be running
    _free_ports(service)
    if preserve_volumes:
        _run_compose(f"up -d {service}")
    else:
        _run_compose(f"up -d -V {service}")
    time.sleep(5)
    print(f"  {service} container is up.")


def stop_container(service: str):
    """Stop and remove a single service container and its volumes."""
    print(f"\n  Stopping {service} container …")
    _run_compose(f"rm -fsv {service}")
    print(f"  {service} container stopped and removed.")


def stop_all_containers():
    """Stop all benchmark containers and remove volumes."""
    print("\n  Stopping all benchmark containers …")
    try:
        _run_compose("down -v", timeout=60)
    except Exception:
        pass
    print("  All containers stopped.")


def get_container_memory_usage_mb(container_name: str) -> float | None:
    """Get actual memory usage of a Docker container in MB via Docker API."""
    import subprocess
    try:
        result = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container_name],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        # Output format: "1.234GiB / 4GiB" or "567.8MiB / 4GiB"
        usage_str = result.stdout.strip().split("/")[0].strip()
        if "GiB" in usage_str:
            return float(usage_str.replace("GiB", "").strip()) * 1024
        elif "MiB" in usage_str:
            return float(usage_str.replace("MiB", "").strip())
        elif "KiB" in usage_str:
            return float(usage_str.replace("KiB", "").strip()) / 1024
        return None
    except Exception:
        return None
