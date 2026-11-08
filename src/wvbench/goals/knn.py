"""Goal 1 - KNN: db 1 <query_id> <k>

Nearest neighbours of test vector <query_id>. Prints one result per line as
`id distance`, nearest first. ids are 0-based train rows decoded from the
object UUID, so no property is read.
"""

from __future__ import annotations

from weaviate.classes.query import MetadataQuery

from wvbench import client as wclient
from wvbench import dataset as ds
from wvbench.settings import ConfigError, ProfileConfig

USAGE = "db 1 <query_id> <k>"


def run(params: list[str], dataset_name: str) -> list[tuple[int, float]]:
    if len(params) != 2:
        raise ConfigError(f"usage: {USAGE}")
    try:
        query_id, k = int(params[0]), int(params[1])
    except ValueError as e:
        raise ConfigError(f"usage: {USAGE} (integers expected)") from e
    if k < 1:
        raise ConfigError("k must be >= 1")

    info = ds.info(dataset_name)
    if not 0 <= query_id < (info.n_queries or 0):
        raise ConfigError(f"query_id {query_id} out of range 0..{(info.n_queries or 0) - 1}")
    vector = ds.read_rows(info.path, "test", [query_id])[0]
    vector_name = ProfileConfig().raw["schema"]["vector_name"]

    client = wclient.connect(wait_s=10)
    try:
        if not client.collections.exists(info.collection):
            raise ConfigError(f"collection {info.collection} does not exist; load {dataset_name} first")
        res = client.collections.use(info.collection).query.near_vector(
            near_vector=vector,
            target_vector=vector_name,
            limit=k,
            return_properties=[],
            return_metadata=MetadataQuery(distance=True),
        )
    finally:
        client.close()
    return [(ds.uuid_row(o.uuid), float(o.metadata.distance)) for o in res.objects]
