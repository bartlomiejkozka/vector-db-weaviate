"""Collection DDL in the two parts required by the Common Baseline.

SCHEMA PART: the collection with the two required columns
    id     -> object UUID = UUID(int=row) (+ INT property `row_id` when schema.store_row_id = true)
    vector -> one self-provided named vector, float32[d]
INDEX PART: the vector index with its type and every parameter.

Weaviate builds the vector index while objects are inserted, so the index
definition has to be given when the collection is created (baseline DDL
exception). With ASYNC_INDEXING=true the graph is built from a queue, which lets
us time loading and index building separately (see timing.mode).
"""

from __future__ import annotations

from typing import Any

import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances

from wvbench import sizing
from wvbench.resolve import Resolved

DISTANCES = {
    "cosine": VectorDistances.COSINE,
    "l2-squared": VectorDistances.L2_SQUARED,
    "dot": VectorDistances.DOT,
}


def _opt(value: Any) -> Any:
    """-1 / 0 in the config means 'use the Weaviate default' for optional parameters."""
    return None if value in (-1, 0, None) else value


# =============================================================================
# === SCHEMA PART =============================================================
# =============================================================================

def schema_definition(r: Resolved) -> dict[str, Any]:
    """Human-readable schema, printed by scripts/ddl.py and saved in the run log."""
    sch = r.section("schema")
    cols = [{"column": "id", "weaviate": "object UUID = UUID(int=row)", "type": "64-bit integer (encoded)"}]
    if sch["store_row_id"]:
        cols.append({"column": "id", "weaviate": "property row_id", "type": "INT (int64)",
                     "filterable": bool(sch["row_id_filterable"])})
    cols.append({"column": "vector", "weaviate": f"named vector {sch['vector_name']!r}",
                 "type": f"float32[{r.info.dim}]"})
    return {"collection": r.info.collection, "baseline_table": r.info.table, "columns": cols,
            "shards": 1, "replication_factor": 1}


def properties(r: Resolved) -> list[Property]:
    if not r.section("schema")["store_row_id"]:
        return []
    return [Property(name="row_id", data_type=DataType.INT,
                     index_filterable=bool(r.section("schema")["row_id_filterable"]),
                     index_searchable=False, index_range_filters=False)]


def create_collection(client: weaviate.WeaviateClient, r: Resolved) -> None:
    """Drop and recreate the collection (re-running the DDL gives a clean table)."""
    name = r.info.collection
    if client.collections.exists(name):
        client.collections.delete(name)
    client.collections.create(
        name=name,
        description=f"Common Baseline: {r.info.name} ({r.info.rows} x {r.info.dim})",
        properties=properties(r),
        vector_config=Configure.Vectors.self_provided(
            name=r.section("schema")["vector_name"],
            vector_index_config=vector_index_config(r),   # INDEX PART
        ),
        sharding_config=Configure.sharding(desired_count=1),
        replication_config=Configure.replication(factor=1),
    )


# =============================================================================
# === INDEX PART ==============================================================
# =============================================================================

def index_definition(r: Resolved) -> dict[str, Any]:
    """The parameters that are actually sent to Weaviate (for the README / run log)."""
    idx = r.index
    kind = idx["index"]
    out: dict[str, Any] = {"type": kind, "distance": r.info.metric if idx["distance"] == "auto" else idx["distance"]}
    if kind == "hnsw":
        for k in ("max_connections", "ef_construction", "ef", "dynamic_ef_min", "dynamic_ef_max",
                  "dynamic_ef_factor", "flat_search_cutoff", "vector_cache_max_objects", "cleanup_interval_seconds"):
            out[k] = idx[k]
    elif kind == "flat":
        out["vector_cache_max_objects"] = idx["vector_cache_max_objects"]
    elif kind == "hfresh":
        for k in ("max_posting_size_kb", "replicas", "search_probe"):
            out[k] = idx[k]
    q = "rq" if kind == "hfresh" else idx["quantizer"]
    out["quantizer"] = q
    if q == "rq":
        out.update(rq_bits=1 if kind == "hfresh" else idx["rq_bits"], rescore_limit=idx["rescore_limit"])
    elif q == "pq":
        out.update(pq_segments=sizing.pq_segments(idx, r.info.dim), pq_centroids=idx["pq_centroids"],
                   training_limit=idx["training_limit"])
    elif q == "sq":
        out.update(rescore_limit=idx["rescore_limit"], training_limit=idx["training_limit"])
    elif q == "bq":
        out.update(rescore_limit=idx["rescore_limit"])
    return out


def quantizer_config(r: Resolved):
    idx = r.index
    q = "rq" if idx["index"] == "hfresh" else idx["quantizer"]
    Q = Configure.VectorIndex.Quantizer
    if q == "none":
        return Q.none() if idx["index"] == "hnsw" else None
    if q == "rq":
        bits = 1 if idx["index"] == "hfresh" else int(idx["rq_bits"])
        return Q.rq(bits=bits, rescore_limit=_opt(idx["rescore_limit"]))
    if q == "pq":
        return Q.pq(segments=sizing.pq_segments(idx, r.info.dim), centroids=idx["pq_centroids"],
                    training_limit=idx["training_limit"])
    if q == "sq":
        return Q.sq(rescore_limit=_opt(idx["rescore_limit"]), training_limit=idx["training_limit"])
    if q == "bq":
        return Q.bq(rescore_limit=_opt(idx["rescore_limit"]))
    raise ValueError(f"unknown quantizer {q!r}")


def vector_index_config(r: Resolved):
    idx = r.index
    metric = r.info.metric if idx["distance"] == "auto" else idx["distance"]
    distance = DISTANCES[metric]
    VI = Configure.VectorIndex
    kind = idx["index"]
    if kind == "hnsw":
        return VI.hnsw(
            distance_metric=distance,
            max_connections=idx["max_connections"],
            ef_construction=idx["ef_construction"],
            ef=idx["ef"],
            dynamic_ef_min=idx["dynamic_ef_min"],
            dynamic_ef_max=idx["dynamic_ef_max"],
            dynamic_ef_factor=idx["dynamic_ef_factor"],
            flat_search_cutoff=idx["flat_search_cutoff"],
            vector_cache_max_objects=int(idx["vector_cache_max_objects"]),
            cleanup_interval_seconds=idx["cleanup_interval_seconds"],
            quantizer=quantizer_config(r),
        )
    if kind == "flat":
        return VI.flat(distance_metric=distance,
                       vector_cache_max_objects=int(idx["vector_cache_max_objects"]),
                       quantizer=quantizer_config(r))
    if kind == "hfresh":
        return VI.hfresh(distance_metric=distance, max_posting_size_kb=idx["max_posting_size_kb"],
                         replicas=idx["replicas"], search_probe=idx["search_probe"],
                         quantizer=quantizer_config(r))
    raise ValueError(f"unknown index type {kind!r}")
