"""Measurements: container memory peak, Weaviate RSS sampling, on-disk sizes, system info.

Peak memory (baseline: "peak resident memory or container memory peak"):
  * primary: cgroup of the Weaviate container (memory.peak on cgroup v2,
    memory.max_usage_in_bytes on v1), read through the Docker socket. The
    container is restarted before loading, so the peak starts from zero; it
    is monotonic, so reading it after load / index / search gives the peak up
    to that phase. Note: the cgroup also counts page cache of the container.
  * auxiliary: Weaviate's Prometheus endpoint (process_resident_memory_bytes),
    sampled in a background thread ONLY during load and index build. During
    search it is read once before and once after, because scraping /metrics
    while timing single queries would distort p99.
"""

from __future__ import annotations

import os
import platform
import re
import threading
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import requests

from wvbench import settings

MB = 1e6


# --------------------------------------------------------------------------
# Docker / cgroup
# --------------------------------------------------------------------------
class WeaviateContainer:
    """Access to the Weaviate container through the Docker socket (optional)."""

    def __init__(self, name: str = settings.WEAVIATE_CONTAINER):
        self.name = name
        self.error: str | None = None
        self._container = None
        try:
            import docker

            self._container = docker.from_env(timeout=30).containers.get(name)
        except Exception as e:  # no socket, no SDK, or container not found
            self.error = f"{type(e).__name__}: {e}"

    @property
    def available(self) -> bool:
        return self._container is not None

    def _exec(self, cmd: str) -> str | None:
        res = self._container.exec_run(["sh", "-c", cmd])
        return res.output.decode().strip() if res.exit_code == 0 else None

    def peak_bytes(self) -> int | None:
        if not self.available:
            return None
        out = self._exec("cat /sys/fs/cgroup/memory.peak 2>/dev/null "
                         "|| cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes")
        return int(out) if out and out.isdigit() else None

    def current_bytes(self) -> int | None:
        if not self.available:
            return None
        out = self._exec("cat /sys/fs/cgroup/memory.current 2>/dev/null "
                         "|| cat /sys/fs/cgroup/memory/memory.usage_in_bytes")
        return int(out) if out and out.isdigit() else None

    def restart(self) -> None:
        self._container.restart(timeout=60)
        self._container.reload()

    def env(self) -> dict[str, str]:
        if not self.available:
            return {}
        self._container.reload()
        pairs = (e.split("=", 1) for e in self._container.attrs["Config"]["Env"])
        return {k: v for k, v in pairs}

    def limits(self) -> dict[str, Any]:
        if not self.available:
            return {}
        hc = self._container.attrs["HostConfig"]
        return {
            "cpuset": hc.get("CpusetCpus"),
            "memory_bytes": hc.get("Memory"),
            "memory_swap_bytes": hc.get("MemorySwap"),
            "image": self._container.attrs["Config"]["Image"],
            "oom_killed": self._container.attrs["State"].get("OOMKilled"),
        }


# --------------------------------------------------------------------------
# Prometheus RSS sampler
# --------------------------------------------------------------------------
_RSS_RE = re.compile(r"^process_resident_memory_bytes\s+([0-9.eE+]+)", re.M)


def weaviate_rss_bytes(timeout: float = 2.0) -> int | None:
    url = f"http://{settings.WEAVIATE_HOST}:{settings.WEAVIATE_METRICS_PORT}/metrics"
    try:
        m = _RSS_RE.search(requests.get(url, timeout=timeout).text)
        return int(float(m.group(1))) if m else None
    except requests.RequestException:
        return None


class RssSampler:
    """Background sampling of Weaviate RSS. Use only around load / index, never around search."""

    def __init__(self, interval_s: float = 0.5, enabled: bool = True):
        self.interval_s = interval_s
        self.enabled = enabled
        self.peak: int | None = None
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            v = weaviate_rss_bytes()
            if v is not None:
                self.samples += 1
                self.peak = v if self.peak is None else max(self.peak, v)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "RssSampler":
        if self.enabled:
            self._thread = threading.Thread(target=self._run, name="rss-sampler", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


class MemoryProbe:
    """Peak memory according to metrics.memory_source."""

    def __init__(self, metrics_cfg: dict[str, Any]):
        self.cfg = metrics_cfg
        self.source = metrics_cfg.get("memory_source", "cgroup")
        self.container = WeaviateContainer() if self.source == "cgroup" else None
        if self.container is not None and not self.container.available:
            print(f"warning: memory_source=cgroup but the Docker socket is not usable "
                  f"({self.container.error}); falling back to prometheus", flush=True)
            self.source = "prometheus"

    def sampler(self) -> RssSampler:
        return RssSampler(float(self.cfg.get("sample_interval_s", 0.5)),
                          enabled=bool(self.cfg.get("prometheus_sampling", True)) and self.source != "off")

    def peak_mb(self, sampler: RssSampler | None = None) -> float | None:
        """Peak so far: cgroup peak, or the sampler peak for the prometheus source."""
        if self.source == "cgroup":
            v = self.container.peak_bytes()
            return round(v / MB, 1) if v is not None else None
        if self.source == "prometheus":
            vals = [x for x in (sampler.peak if sampler else None, weaviate_rss_bytes()) if x is not None]
            return round(max(vals) / MB, 1) if vals else None
        return None


# --------------------------------------------------------------------------
# On-disk sizes (the Weaviate volume is mounted read-only at /db in the client)
# --------------------------------------------------------------------------
INDEX_PATTERNS = ("*.hnsw*", "*hnsw*", "*hfresh*", "*vectors_compressed*", "*flat*")


def dir_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def collection_dir(collection: str, db_dir: Path = settings.DB_DIR) -> Path | None:
    """Weaviate stores a collection under <data path>/<lowercase collection name>/."""
    p = db_dir / collection.lower()
    return p if p.is_dir() else None


def index_bytes(coll_dir: Path) -> int:
    """Vector index files: HNSW commit logs and snapshots, compressed vectors, HFresh/flat data."""
    seen: set[Path] = set()
    total = 0
    for pattern in INDEX_PATTERNS:
        for p in coll_dir.rglob(pattern):
            if any(parent in seen for parent in p.parents) or p in seen:
                continue
            seen.add(p)
            total += dir_bytes(p) if p.is_dir() else p.stat().st_size
    return total


def disk_sizes_mb(collection: str) -> dict[str, float | None]:
    db = settings.DB_DIR
    if not db.is_dir():
        return {"index_mb": None, "db_mb": None}
    cdir = collection_dir(collection, db)
    return {
        "index_mb": round(index_bytes(cdir) / MB, 1) if cdir else None,
        "db_mb": round(dir_bytes(db) / MB, 1),
    }


# --------------------------------------------------------------------------
# System info (logged with every result)
# --------------------------------------------------------------------------
def _cpu_model() -> str:
    if os.getenv("HOST_CPU_MODEL"):
        return os.environ["HOST_CPU_MODEL"]
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith(("model name", "hardware")):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _cores(cpuset: str | None) -> int | None:
    if not cpuset:
        return None
    n = 0
    for part in cpuset.split(","):
        a, _, b = part.partition("-")
        n += int(b) - int(a) + 1 if b else 1
    return n


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def system_info(container: WeaviateContainer | None = None) -> dict[str, Any]:
    limits = container.limits() if container and container.available else {}
    cpuset = limits.get("cpuset") or os.getenv("WEAVIATE_CPUS")
    mem = limits.get("memory_bytes")
    return {
        "cpu": _cpu_model(),
        "cores": _cores(cpuset),
        "cpuset": cpuset,
        "ram_gb": round(mem / 2**30, 2) if mem else os.getenv("WEAVIATE_MEM_LIMIT"),
        "host_ram_gb": os.getenv("HOST_RAM_GB"),
        "host_os": os.getenv("HOST_OS") or platform.platform(),
        "disk": os.getenv("HOST_DISK_TYPE", "unknown"),
        "container": limits,
        "python": platform.python_version(),
        "libraries": {p: _version(p) for p in ("weaviate-client", "h5py", "numpy", "docker", "requests")},
    }
