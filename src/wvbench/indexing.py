"""Index build timing.

timing.mode:
  async - ASYNC_INDEXING=true. Objects are stored first and queued for the
          vector index. build_s = time from the end of loading until the queue
          of every shard is empty and no shard is INDEXING. Note: indexing of
          the queue already starts while loading runs.
  sync  - ASYNC_INDEXING=false. The index is built during the insert, so
          ingest_s = load + index and build_s stays empty (baseline DDL exception).
  auto  - take the mode from ASYNC_INDEXING (the same .env value is given to
          the Weaviate and the client container).
"""

from __future__ import annotations

import os
import sys
import time

import weaviate

from wvbench.settings import ConfigError


def _truthy(v: str | None) -> bool | None:
    if v is None or v == "":
        return None
    return v.strip().lower() in ("1", "true", "yes", "on")


def timing_mode(timing_cfg: dict, server_env: dict[str, str] | None = None) -> str:
    """Resolve timing.mode and check it against the server's ASYNC_INDEXING."""
    server = _truthy((server_env or {}).get("ASYNC_INDEXING"))
    client_side = _truthy(os.getenv("ASYNC_INDEXING"))
    actual = server if server is not None else client_side
    mode = timing_cfg.get("mode", "auto")
    if mode == "auto":
        if actual is None:
            raise ConfigError("timing.mode=auto but ASYNC_INDEXING is unknown; set it in .env")
        return "async" if actual else "sync"
    if mode not in ("async", "sync"):
        raise ConfigError(f"timing.mode must be auto, async or sync, got {mode!r}")
    if actual is not None and actual != (mode == "async"):
        raise ConfigError(
            f"timing.mode={mode} but the server runs with ASYNC_INDEXING={'true' if actual else 'false'}. "
            f"Set ASYNC_INDEXING={'true' if mode == 'async' else 'false'} in .env and recreate the "
            "weaviate container (docker compose up -d weaviate)."
        )
    return mode


def queue_status(client: weaviate.WeaviateClient, collection: str) -> tuple[int, set[str], int]:
    """(vectors waiting in the index queue, shard statuses, stored objects)."""
    nodes = client.cluster.nodes(collection=collection, output="verbose")
    queue, statuses, objects = 0, set(), 0
    for node in nodes:
        for shard in node.shards or []:
            if shard.collection != collection:
                continue
            queue += shard.vector_queue_length
            statuses.add(shard.vector_indexing_status)
            objects += shard.object_count
    return queue, statuses, objects


def wait_until_indexed(client: weaviate.WeaviateClient, collection: str, poll_s: float = 1.0,
                       timeout_s: float = 0, log=print) -> float:
    """Block until the vector queue is empty twice in a row; return the seconds waited."""
    t0 = time.perf_counter()
    done_streak = 0
    last_print = 0.0
    while True:
        queue, statuses, _ = queue_status(client, collection)
        if queue == 0 and "INDEXING" not in statuses:
            done_streak += 1
            if done_streak >= 2:
                # The second check was one poll interval after the first one that saw an
                # empty queue; report the time of the first one.
                return max(0.0, time.perf_counter() - t0 - poll_s)
        else:
            done_streak = 0
        el = time.perf_counter() - t0
        if el - last_print >= 10:
            log(f"  indexing: queue={queue:,} status={','.join(sorted(statuses))} ({el:.0f}s)",
                file=sys.stderr, flush=True)
            last_print = el
        if timeout_s and el > timeout_s:
            raise TimeoutError(f"index queue not empty after {timeout_s}s (queue={queue})")
        time.sleep(poll_s)
