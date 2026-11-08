"""Download step: fetch the dataset file into /data, print its size and sha256."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import requests

from wvbench import dataset as ds
from wvbench.settings import ConfigError

CHUNK = 8 * 1024 * 1024


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def download(name: str) -> Path:
    reg = ds.registry()
    entry = reg.get("datasets", {}).get(name)
    if entry is None:
        raise ConfigError(f"unknown dataset {name!r}; known: {', '.join(reg.get('datasets', {}))}")
    url = entry.get("url") or f"{reg['base_url']}/{name}.hdf5"
    path = ds.dataset_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        print(f"{path} already exists, verifying")
    else:
        tmp = path.with_suffix(".hdf5.part")
        print(f"downloading {url}")
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            done = 0
            with open(tmp, "wb") as f:
                for block in resp.iter_content(CHUNK):
                    f.write(block)
                    done += len(block)
                    if total:
                        print(f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB ({done / total:.0%})",
                              end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
        tmp.rename(path)  # only after a complete download

    size = path.stat().st_size
    digest = sha256(path)
    print(f"file:   {path}")
    print(f"size:   {size} bytes ({size / 1e9:.2f} GB)")
    print(f"sha256: {digest}")
    if entry.get("bytes") and size != entry["bytes"]:
        raise ConfigError(f"size mismatch: expected {entry['bytes']} bytes")
    if entry.get("sha256") and digest != entry["sha256"]:
        raise ConfigError(f"checksum mismatch: expected {entry['sha256']}")
    if entry.get("sha256"):
        print("checksum matches config/datasets.toml")
    return path
