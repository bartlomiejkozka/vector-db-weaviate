"""Post-load check (baseline step 5): row count, dimension, ~10 random ids match the file."""

from __future__ import annotations

import numpy as np
import weaviate

from wvbench import dataset as ds
from wvbench.resolve import Resolved


class CheckError(Exception):
    pass


def run_check(client: weaviate.WeaviateClient, r: Resolved, log=print) -> dict:
    cfg = r.section("check")
    vector_name = r.section("schema")["vector_name"]
    coll = client.collections.use(r.info.collection)

    count = coll.aggregate.over_all(total_count=True).total_count
    problems = []
    if count != r.info.rows:
        problems.append(f"row count {count} != {r.info.rows} train rows")

    rng = np.random.default_rng(int(cfg["seed"]))
    rows = sorted(rng.choice(r.info.rows, size=min(int(cfg["samples"]), r.info.rows), replace=False).tolist())
    expected = ds.read_rows(r.info.path, "train", rows)
    for row, exp in zip(rows, expected):
        obj = coll.query.fetch_object_by_id(ds.row_uuid(row), include_vector=True)
        if obj is None:
            problems.append(f"id {row}: not found")
            continue
        got = np.asarray(obj.vector[vector_name], dtype=np.float32)
        if got.shape != (r.info.dim,):
            problems.append(f"id {row}: dimension {got.shape[0]} != {r.info.dim}")
        elif not np.allclose(got, exp, rtol=1e-6, atol=1e-6):
            problems.append(f"id {row}: vector differs from the file (max abs diff {np.abs(got - exp).max():.3g})")
        if r.section("schema")["store_row_id"] and obj.properties.get("row_id") != row:
            problems.append(f"id {row}: row_id property is {obj.properties.get('row_id')}")

    result = {"count": count, "expected": r.info.rows, "sampled_ids": rows, "problems": problems}
    if problems:
        for p in problems:
            log(f"  CHECK FAILED: {p}")
        raise CheckError(f"{len(problems)} check(s) failed")
    log(f"  check ok: {count:,} rows, dim {r.info.dim}, {len(rows)} random ids match the file")
    return result
