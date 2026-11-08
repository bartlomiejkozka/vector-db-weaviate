"""Baseline loading order: schema -> load -> index -> check, timed separately, then build.csv.

Each step can run alone (scripts/ddl.py, load.py, index.py, check.py) and keeps
its measurements in results/runs/<dataset>.state.json; scripts/build.py runs all
of them and writes the build.csv row.
"""

from __future__ import annotations

import sys
from typing import Any

from wvbench import check, client as wclient, indexing, ingest, metrics, results, schema
from wvbench.resolve import Resolved


def log(*args, **kwargs) -> None:
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


class Run:
    def __init__(self, r: Resolved):
        self.r = r
        self.state = results.RunState(r.info.name)
        self.probe = metrics.MemoryProbe(r.section("metrics"))
        self.client = None

    @property
    def container(self) -> metrics.WeaviateContainer | None:
        return self.probe.container

    def connect(self):
        if self.client is None:
            self.client = wclient.connect()
        return self.client

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def server_env(self) -> dict[str, str]:
        return self.container.env() if self.container and self.container.available else {}

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def step_schema(run: Run) -> None:
    r = run.r
    client = run.connect()
    # Drop first, then restart: a restart with the old collection would load it and inflate the peak.
    if client.collections.exists(r.info.collection):
        log(f"dropping existing collection {r.info.collection}")
        client.collections.delete(r.info.collection)
    if r.section("metrics").get("restart_before_load") and run.container and run.container.available:
        log("restarting Weaviate so the container memory peak starts from zero")
        run.close()
        run.container.restart()
        client = run.connect()
    mode = indexing.timing_mode(r.section("timing"), run.server_env())
    schema.create_collection(client, r)
    log(f"created collection {r.info.collection} ({r.index['index']}, profile {r.settings.profile})")
    run.state.reset(
        config=r.as_dict(),
        schema={"definition": schema.schema_definition(r), "index": schema.index_definition(r),
                "timing_mode": mode, "server_version": wclient.server_version(client),
                "server_env": run.server_env()},
    )


def step_load(run: Run) -> ingest.IngestResult:
    r = run.r
    client = run.connect()
    if not client.collections.exists(r.info.collection):
        raise RuntimeError(f"collection {r.info.collection} does not exist; run the schema step first")
    log(f"loading {r.info.rows:,} x {r.info.dim} from {r.info.path} (ingest.mode={r.section('ingest')['mode']})")
    with run.probe.sampler() as sampler:
        res = ingest.load(client, r, log=log)
    peak = run.probe.peak_mb(sampler)
    log(f"loaded {res.rows:,} rows in {res.seconds:.1f}s, peak memory {peak} MB")
    run.state.update("load", load_s=round(res.seconds, 3), rows=res.rows, failed=res.failed, mode=res.mode,
                     peak_mem_mb=peak, rss_peak_mb=_mb(sampler.peak))
    return res


def step_index(run: Run) -> dict[str, Any]:
    r = run.r
    client = run.connect()
    mode = indexing.timing_mode(r.section("timing"), run.server_env())
    t = r.section("timing")
    log(f"waiting for the vector index (timing.mode={mode})")
    with run.probe.sampler() as sampler:
        waited = indexing.wait_until_indexed(client, r.info.collection, float(t["poll_interval_s"]),
                                             float(t["index_timeout_s"]), log=log)
    peak = run.probe.peak_mb(sampler)
    sizes = metrics.disk_sizes_mb(r.info.collection)
    load_s = run.state.get("load", "load_s")
    if mode == "async":
        out = {"ingest_s": load_s, "build_s": round(waited, 3)}
    else:  # DDL exception: one ingestion time, build_s empty
        out = {"ingest_s": round((load_s or 0) + waited, 3), "build_s": None}
    out.update(timing_mode=mode, wait_s=round(waited, 3), peak_mem_mb=peak, rss_peak_mb=_mb(sampler.peak), **sizes)
    log(f"index ready: ingest_s={out['ingest_s']} build_s={out['build_s']} "
        f"index_mb={sizes['index_mb']} db_mb={sizes['db_mb']} peak={peak} MB")
    run.state.update("index", **out)
    return out


def step_check(run: Run) -> dict[str, Any]:
    res = check.run_check(run.connect(), run.r, log=log)
    run.state.update("check", **res)
    return res


def finalize(run: Run) -> dict[str, Any]:
    """Write the build.csv row and the full run log."""
    r, st = run.r, run.state
    info = metrics.system_info(run.container)
    peaks = [p for p in (st.get("load", "peak_mem_mb"), st.get("index", "peak_mem_mb")) if p is not None]
    row = {
        "group": r.section("run")["group"],
        "system": r.section("run")["system"],
        "version": st.get("schema", "server_version"),
        "dataset": r.info.name,
        "ingest_s": st.get("index", "ingest_s"),
        "build_s": st.get("index", "build_s"),
        "index_mb": st.get("index", "index_mb"),
        "db_mb": st.get("index", "db_mb"),
        "peak_mem_mb": max(peaks) if peaks else None,
        "cpu": info["cpu"],
        "cores": info["cores"],
        "ram_gb": info["ram_gb"],
        "date": results.now().date().isoformat(),
        "kmeans_s": None,
    }
    csv_path = results.append_csv("build.csv", results.BUILD_COLUMNS, row)
    log_path = results.write_run_log(r.info.name, "build", {"row": row, "system": info, **st.data})
    log(f"appended {csv_path}\nrun log {log_path}")
    return row


def _mb(v: int | None) -> float | None:
    return round(v / metrics.MB, 1) if v is not None else None


def fail(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)
