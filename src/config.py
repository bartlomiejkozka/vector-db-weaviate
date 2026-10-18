"""Project configuration loaded from config.toml (repository root)."""

import os
import tomllib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

with open(ROOT_DIR / "config.toml", "rb") as _f:
    _cfg = tomllib.load(_f)

# Weaviate connection (env vars override config.toml)
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", _cfg["weaviate"]["host"])
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_HTTP_PORT", _cfg["weaviate"]["http_port"]))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", _cfg["weaviate"]["grpc_port"]))

# Dataset
DATASET_NAME = _cfg["dataset"]["name"]
DATASET_URL = _cfg["dataset"]["url"]
DATA_DIR = ROOT_DIR / _cfg["dataset"]["data_dir"]
DATASET_PATH = DATA_DIR / f"{DATASET_NAME}.hdf5"

# Collection
COLLECTION_NAME = _cfg["collection"]["name"]

# Import
BATCH_SIZE = _cfg["import"]["batch_size"]
READ_CHUNK = _cfg["import"]["read_chunk"]
