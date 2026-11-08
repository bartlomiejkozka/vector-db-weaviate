"""VIBE HDF5 dataset access: metadata, chunked train reading, queries and ground truth."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from wvbench import settings
from wvbench.settings import ConfigError

# File distance attribute -> Weaviate distance metric.
# "normalized"/"any": the baseline says to use cosine.
METRICS = {
    "cosine": "cosine",
    "angular": "cosine",
    "normalized": "cosine",
    "any": "cosine",
    "euclidean": "l2-squared",
    "l2": "l2-squared",
    "ip": "dot",
    "dot": "dot",
}


def registry() -> dict[str, Any]:
    return settings.load_toml("datasets.toml")


def dataset_path(name: str) -> Path:
    return settings.DATA_DIR / f"{name}.hdf5"


def table_name(name: str) -> str:
    """Baseline table name: file name without extension, '-' -> '_'."""
    return name.replace("-", "_")


def collection_name(name: str) -> str:
    """Weaviate always capitalizes the first letter of a collection name."""
    t = table_name(name)
    return t[:1].upper() + t[1:]


def row_uuid(row: int) -> uuid.UUID:
    """Object UUID that encodes the 0-based train row (reversible, no property lookup)."""
    return uuid.UUID(int=row)


def uuid_row(u: uuid.UUID | str) -> int:
    return (u if isinstance(u, uuid.UUID) else uuid.UUID(str(u))).int


@dataclass
class DatasetInfo:
    name: str
    path: str
    source: str           # "file" or "registry" (file not downloaded yet)
    rows: int
    dim: int
    distance: str         # as written in the file
    metric: str           # Weaviate metric
    point_type: str
    n_queries: int | None
    gt_k: int | None

    @property
    def table(self) -> str:
        return table_name(self.name)

    @property
    def collection(self) -> str:
        return collection_name(self.name)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(table=self.table, collection=self.collection)
        return d


def _metric(distance: str) -> str:
    try:
        return METRICS[distance.lower()]
    except KeyError as e:
        raise ConfigError(f"unsupported distance {distance!r} in dataset file") from e


def _attr(f, key: str, default: Any = None) -> Any:
    v = f.attrs.get(key, default)
    return v.decode() if isinstance(v, bytes) else v


def info(name: str, require_file: bool = True) -> DatasetInfo:
    """Read dataset metadata from the file, or from the registry if allowed and not downloaded."""
    reg = registry().get("datasets", {}).get(name)
    path = dataset_path(name)
    if not path.exists():
        if require_file or reg is None:
            hint = "" if reg else f" (and {name!r} is not in config/datasets.toml)"
            raise ConfigError(f"dataset file not found: {path}{hint}. Run the download step first.")
        return DatasetInfo(name, str(path), "registry", reg["rows"], reg["dim"], reg["distance"],
                           _metric(reg["distance"]), "float", None, None)

    import h5py

    with h5py.File(path, "r") as f:
        for key in ("train", "test", "neighbors"):
            if key not in f:
                raise ConfigError(f"{path}: missing dataset {key!r}")
        rows, dim = f["train"].shape
        distance = str(_attr(f, "distance", reg["distance"] if reg else "cosine"))
        point_type = str(_attr(f, "point_type", "float"))
        result = DatasetInfo(name, str(path), "file", int(rows), int(dim), distance, _metric(distance),
                             point_type, int(f["test"].shape[0]), int(f["neighbors"].shape[1]))
    if reg and (reg["rows"], reg["dim"]) != (result.rows, result.dim):
        raise ConfigError(
            f"{name}: file has {result.rows}x{result.dim}, registry says {reg['rows']}x{reg['dim']}"
        )
    return result


def to_float32(chunk: np.ndarray) -> np.ndarray:
    """The only conversion the baseline allows: uint8 -> float32."""
    if chunk.dtype == np.float32:
        return chunk
    if chunk.dtype == np.uint8:
        return chunk.astype(np.float32)
    raise ConfigError(f"unexpected vector dtype {chunk.dtype}; only float32 and uint8 are allowed")


def iter_train(path: str | Path, chunk_rows: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (first_row, float32 array) chunks so the whole file never has to fit in memory."""
    import h5py

    with h5py.File(path, "r") as f:
        train = f["train"]
        for start in range(0, train.shape[0], chunk_rows):
            yield start, to_float32(train[start:start + chunk_rows])


def read_rows(path: str | Path, key: str, rows: list[int]) -> np.ndarray:
    import h5py

    order = np.argsort(rows)
    with h5py.File(path, "r") as f:
        data = f[key][np.asarray(rows)[order]]  # h5py needs increasing indices
    out = np.empty_like(data)
    out[order] = data
    return to_float32(out) if key in ("train", "test") else out


def queries(path: str | Path) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as f:
        return to_float32(f["test"][:])


def ground_truth(path: str | Path) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as f:
        return f["neighbors"][:]
