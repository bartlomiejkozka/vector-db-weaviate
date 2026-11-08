"""Bulk load of the train vectors.

The file is always read in chunks of ingest.read_chunk rows (baseline: ~50,000).
How the objects are sent to Weaviate is chosen by ingest.mode:
  client_fixed   - client-side batching with a fixed batch size and concurrency
  client_dynamic - client-side batching, the client adapts the batch size
  server_side    - server-side batching (Weaviate >= 1.36): one gRPC stream,
                   the server tells the client how many objects to send next
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import weaviate

from wvbench import dataset as ds
from wvbench.resolve import Resolved

MODES = ("client_fixed", "client_dynamic", "server_side")


class IngestError(Exception):
    pass


@dataclass
class IngestResult:
    rows: int
    seconds: float
    failed: int
    mode: str


def _batch_context(collection, cfg: dict):
    mode = cfg["mode"]
    if mode == "client_fixed":
        return collection.batch.fixed_size(batch_size=int(cfg["batch_size"]),
                                           concurrent_requests=int(cfg["concurrent_requests"]))
    if mode == "client_dynamic":
        return collection.batch.dynamic()
    if mode == "server_side":
        return collection.batch.stream(concurrency=int(cfg["stream_concurrency"]))
    raise IngestError(f"ingest.mode must be one of {MODES}, got {mode!r}")


def load(client: weaviate.WeaviateClient, r: Resolved, log=print) -> IngestResult:
    cfg = r.section("ingest")
    store_row_id = bool(r.section("schema")["store_row_id"])
    vector_name = r.section("schema")["vector_name"]
    max_errors = int(cfg.get("max_errors", 0))
    collection = client.collections.use(r.info.collection)
    total = r.info.rows
    loaded = 0

    t0 = time.perf_counter()
    with _batch_context(collection, cfg) as batch:
        for start, chunk in ds.iter_train(r.info.path, int(cfg["read_chunk"])):
            for i, vec in enumerate(chunk):
                row = start + i
                batch.add_object(
                    properties={"row_id": row} if store_row_id else {},
                    uuid=ds.row_uuid(row),
                    vector={vector_name: vec},
                )
            loaded += len(chunk)
            if batch.number_errors > max_errors:
                break
            el = time.perf_counter() - t0
            log(f"  loaded {loaded:>10,} / {total:,}  ({loaded / el:,.0f} rows/s)", file=sys.stderr, flush=True)
    seconds = time.perf_counter() - t0

    failed = collection.batch.failed_objects
    if len(failed) > max_errors:
        for err in failed[:5]:
            log(f"  error: {err.message}", file=sys.stderr)
        raise IngestError(f"{len(failed)} objects failed to import (ingest.max_errors={max_errors})")
    return IngestResult(rows=loaded, seconds=seconds, failed=len(failed), mode=cfg["mode"])
