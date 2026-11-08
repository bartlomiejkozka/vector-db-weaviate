"""Memory estimate and automatic profile selection (pure functions, no I/O)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

GB = 1e9
BYTES_PER_FLOAT = 4


def raw_bytes(rows: int, dim: int) -> int:
    """Raw float32 size, as defined in the Common Baseline (rows x dim x 4 B)."""
    return rows * dim * BYTES_PER_FLOAT


def pq_segments(index: dict[str, Any], dim: int) -> int:
    seg = int(index.get("pq_segments") or 0)
    return seg if seg > 0 else max(1, dim // 8)


def compression_factor(index: dict[str, Any], dim: int) -> float:
    """How many times smaller the in-memory vectors are than raw float32."""
    if index["index"] == "hfresh":
        return 32.0  # RQ 1-bit is mandatory for HFresh
    q = index.get("quantizer", "none")
    if q == "none":
        return 1.0
    if q == "sq":
        return 4.0
    if q == "bq":
        return 32.0
    if q == "rq":
        return {8: 4.0, 4: 8.0, 1: 32.0}[int(index.get("rq_bits", 8))]
    if q == "pq":
        return dim * BYTES_PER_FLOAT / pq_segments(index, dim)  # one byte per segment
    raise ValueError(f"unknown quantizer {q!r}")


@dataclass
class Estimate:
    mode: str
    raw_bytes: int
    vector_bytes: float
    graph_bytes: float
    total_bytes: float
    budget_bytes: float
    fits: bool
    formula: str

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("raw_bytes", "vector_bytes", "graph_bytes", "total_bytes", "budget_bytes"):
            d[k.replace("_bytes", "_mb")] = round(d.pop(k) / 1e6, 1)
        return d

    def summary(self) -> str:
        verdict = "fits" if self.fits else "DOES NOT FIT"
        return (
            f"[{self.mode}] {self.formula} = {self.total_bytes / GB:.2f} GB "
            f"vs budget {self.budget_bytes / GB:.2f} GB -> {verdict}"
        )


def estimate(rows: int, dim: int, index: dict[str, Any], est: dict[str, Any]) -> Estimate:
    """Estimate peak memory of a configuration.

    vector_bytes: vectors held in RAM = min(rows, vector_cache_max_objects) x dim x 4 / compression
                  (HFresh: rows x replicas x dim x 4 / 32, postings may hold a vector up to `replicas` times)
    graph_bytes:  HNSW layer 0 = rows x 2 x max_connections x bytes_per_link (0 for flat / hfresh)

    simple:   vector_bytes x simple_factor <= memory_limit
    detailed: (vector_bytes + graph_bytes) x overhead_factor <= memory_limit x budget_fraction
    """
    raw = raw_bytes(rows, dim)
    kind = index["index"]
    comp = compression_factor(index, dim)
    if kind == "hfresh":
        vectors = rows * int(index.get("replicas", 4)) * dim * BYTES_PER_FLOAT / comp
    else:
        cached = min(rows, int(index.get("vector_cache_max_objects", rows)))
        vectors = cached * dim * BYTES_PER_FLOAT / comp
    graph = 0.0
    if kind == "hnsw":
        graph = rows * 2 * int(index["max_connections"]) * float(est.get("bytes_per_link", 10))

    limit = float(est.get("memory_limit_gb", 8.0)) * GB
    mode = est.get("mode", "detailed")
    if mode == "simple":
        factor = float(est.get("simple_factor", 1.5))
        total = vectors * factor
        budget = limit
        formula = f"vectors {vectors / GB:.2f} GB x {factor}"
    elif mode == "detailed":
        factor = float(est.get("overhead_factor", 1.25))
        total = (vectors + graph) * factor
        budget = limit * float(est.get("budget_fraction", 0.85))
        formula = f"(vectors {vectors / GB:.2f} GB + graph {graph / GB:.2f} GB) x {factor}"
    else:
        raise ValueError(f"estimator.mode must be 'simple' or 'detailed', got {mode!r}")
    return Estimate(mode, raw, vectors, graph, total, budget, total <= budget, formula)


def _rule_matches(rule: dict[str, Any], rows: int, dim: int) -> bool:
    raw_gb = raw_bytes(rows, dim) / GB
    checks = {
        "min_raw_gb": lambda v: raw_gb >= v,
        "max_raw_gb": lambda v: raw_gb <= v,
        "min_dim": lambda v: dim >= v,
        "max_dim": lambda v: dim <= v,
        "min_rows": lambda v: rows >= v,
        "max_rows": lambda v: rows <= v,
    }
    for key, value in rule.items():
        if key == "profile":
            continue
        if key not in checks:
            raise ValueError(f"unknown selection rule condition {key!r}")
        if not checks[key](value):
            return False
    return True


def select_profile(rules: list[dict[str, Any]], rows: int, dim: int) -> tuple[str | None, str]:
    """Return (profile, reason) of the first matching rule, or (None, reason) when nothing matches."""
    raw_gb = raw_bytes(rows, dim) / GB
    for i, rule in enumerate(rules):
        if _rule_matches(rule, rows, dim):
            conds = ", ".join(f"{k}={v}" for k, v in rule.items() if k != "profile")
            return rule["profile"], f"auto: rule #{i + 1} ({conds}) matched raw={raw_gb:.2f} GB, dim={dim}"
    return None, f"auto: no rule matched raw={raw_gb:.2f} GB, dim={dim}"
