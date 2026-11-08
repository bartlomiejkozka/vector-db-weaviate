"""Result files: build.csv / search.csv (Common Baseline format) and per-run JSON logs."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wvbench import settings

BUILD_COLUMNS = ["group", "system", "version", "dataset", "ingest_s", "build_s", "index_mb", "db_mb",
                 "peak_mem_mb", "cpu", "cores", "ram_gb", "date", "kmeans_s"]
SEARCH_COLUMNS = ["group", "system", "version", "dataset", "k", "setting_name", "setting_value", "recall",
                  "p50_ms", "p95_ms", "p99_ms", "qps", "peak_mem_mb"]


def now() -> datetime:
    return datetime.now(timezone.utc)


def append_csv(name: str, columns: list[str], row: dict[str, Any]) -> Path:
    unknown = set(row) - set(columns)
    if unknown:
        raise ValueError(f"{name}: unknown columns {sorted(unknown)}")
    path = settings.RESULTS_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        if new:
            w.writeheader()
        w.writerow({c: "" if row.get(c) is None else row[c] for c in columns})
    return path


class RunState:
    """Per-dataset state shared by the pipeline steps (results/runs/<dataset>.state.json)."""

    def __init__(self, dataset: str):
        self.path = settings.RESULTS_DIR / "runs" / f"{dataset}.state.json"
        self.data: dict[str, Any] = json.loads(self.path.read_text()) if self.path.exists() else {}

    def reset(self, **values: Any) -> None:
        self.data = dict(values)
        self.save()

    def update(self, phase: str, **values: Any) -> None:
        self.data.setdefault(phase, {}).update(values)
        self.save()

    def get(self, phase: str, key: str, default: Any = None) -> Any:
        return self.data.get(phase, {}).get(key, default)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, default=str))


def write_run_log(dataset: str, kind: str, payload: dict[str, Any]) -> Path:
    path = settings.RESULTS_DIR / "runs" / f"{now():%Y%m%dT%H%M%SZ}_{kind}_{dataset}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
