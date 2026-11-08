# Advanced Database Systems — Vector Database Project (Weaviate)

Benchmark environment for Weaviate under the class's **Common Baseline**: same
VIBE datasets, same resource limits (4 pinned cores, 8 GB RAM, swap off), same
loading order, CLI and result files as every other group.

| Docs checklist | Section |
|---|---|
| 1. Choice of technology | [1](#1-choice-of-technology) |
| 2. Architecture | [2](#2-architecture) |
| 3. Prerequisites | [3](#3-prerequisites) |
| 4. Installation and setup | [4](#4-installation-and-setup) |
| 5. Design and implementation | [5](#5-design-and-implementation) |
| 6. Goals and queries | [6](#6-goals) |
| 7. Roles | [7](#7-team-roles) |
| 8. Results | [8](#8-results) |
| 9. User manual / reproducing results | [9](#9-user-manual) |
| 10. Self-evaluation | [10](#10-self-evaluation) |

## 1. Choice of technology

| Component | Version | Why |
|---|---|---|
| [Weaviate](https://weaviate.io/) | 1.40.1 (`cr.weaviate.io/semitechnologies/weaviate:1.40.1`) | open-source (BSD-3) vector database; HNSW, flat and HFresh indexes, optional quantization (RQ/PQ/BQ/SQ) |
| Python | 3.12.12 (`python:3.12.12-slim-bookworm`) | VIBE files are HDF5, read with `h5py` |
| weaviate-client | 4.23.1 | official client, gRPC for import and search |
| h5py / numpy / requests / docker | 3.16.0 / 2.5.4 / 2.34.2 / 7.1.0 | dataset access, download, container memory peak |
| Docker Compose | v2 | one command per step; Podman with `podman compose` works the same way |

All versions are pinned in [docker-compose.yml](docker-compose.yml), [Dockerfile](Dockerfile) and [requirements.txt](requirements.txt).

## 2. Architecture

Weaviate is a server system, so (Common Baseline §2) the database runs in its
own container with the baseline limits and the client runs in a second
container with its own core.

```
 host: ./data (HDF5)   ./results (CSV, logs)   ./config (TOML/env)
          │ ro                │ rw                   │ ro
┌─────────▼──────────────────▼─────────────────────▼───┐        ┌──────────────────────────────────┐
│ client container (cpuset CLIENT_CPUS, default 4)     │  gRPC  │ weaviate container               │
│  download │ setup (schema→load→index→check) │ db     │───────▶│  cpuset 0-3, mem 8g, swap off    │
│  /data  /results  /db (ro, for sizes)                │  REST  │  GOMAXPROCS=4, GOMEMLIMIT=6800MiB│
│                                                      │───────▶│  /db  (volume weaviate_db)       │
│  docker.sock ── memory.peak / restart ───────────────┼───────▶│  :2112 Prometheus (load/index)   │
└──────────────────────────────────────────────────────┘        └──────────────────────────────────┘
```

| Path in container | Content | Mode |
|---|---|---|
| `/data` | downloaded `<dataset>.hdf5` | read-write in `download`, read-only elsewhere |
| `/db` | Weaviate data (`PERSISTENCE_DATA_PATH`) | read-write in Weaviate, read-only in the client (size measurement) |
| `/results` | `build.csv`, `search.csv`, `runs/*.json` | read-write |

## 3. Prerequisites

- Linux or macOS with Docker (Engine or Desktop) and Docker Compose v2, or Podman with `podman compose`.
- Docker Desktop: the VM needs **≥ 5 CPUs** (4 for Weaviate + 1 for the client) and **~10 GB RAM**.
- Disk: about 3× the raw dataset size (glove 3 GB … arxiv 12 GB) plus the image (~0.5 GB).
- `make` (optional, only a wrapper around `docker compose`).
- Internet access for the image and dataset download.

## 4. Installation and setup

```bash
git clone <repo-url> && cd <repo>
cp .env.example .env            # adjust CPU ids / dataset if needed
docker compose build            # client image
docker compose up -d --wait weaviate
```

## 5. Design and implementation

### 5.1 Repository layout

```
config/weaviate.env     Weaviate server environment (static)
config/datasets.toml    VIBE registry: url, rows, dim, distance, size, sha256
config/profiles.toml    index profiles, selection rules, memory estimator, ingest/timing/metrics
scripts/                download, ddl, load, index, check, build (all steps), show_config, benchmark (M2)
src/wvbench/            implementation (settings, sizing, resolve, dataset, schema, ingest, indexing, check, metrics, results, cli)
tests/                  unit tests of the configuration logic
results/                build.csv, search.csv, runs/*.json (committed)
db                      CLI entry point
```

### 5.2 Database design (DDL)

One collection per dataset. Baseline table name = file name without extension
with `-` → `_`; **Weaviate always capitalises the first letter**, so
`glove_200_cosine` is stored as `Glove_200_cosine`.

| Baseline column | Type | Weaviate | Meaning |
|---|---|---|---|
| `id` | 64-bit integer | object UUID = `UUID(int=row)`; with `schema.store_row_id = true` (default) also the `INT` property `row_id` | 0-based row position in `train`, equals the ids in `neighbors` |
| `vector` | float32, length d | self-provided named vector `vector` | the embedding |

`id` is a reserved name in Weaviate (it is always the UUID), so the row number is
encoded in the UUID. That makes "return ids only" free: queries return no
properties and the id is decoded from the UUID. `row_id` is a plain stored copy.
It has no inverted index by default (`row_id_filterable = false`), because a
filterable index on a unique integer cost 448 MB of disk on glove-200 and nothing
reads it. `store_row_id = false` drops the property entirely.

The DDL lives in [src/wvbench/schema.py](src/wvbench/schema.py) in two marked parts,
`=== SCHEMA PART ===` (collection + columns) and `=== INDEX PART ===` (index type
and every parameter); `scripts/ddl.py` prints both and (re)creates the collection
(re-running drops and recreates it).

**Index after loading.** Weaviate cannot create a vector index on an existing
collection: the index definition is part of the collection and the graph is
built while objects are inserted. This is the baseline's DDL exception; we offer
two timing modes, chosen by `ASYNC_INDEXING` in `.env`:

| Mode | `ASYNC_INDEXING` | `ingest_s` | `build_s` |
|---|---|---|---|
| `async` (default) | `true` | start of loading → last object stored | end of loading → index queue empty on every shard (vectors are queued at insert time and indexing already starts while loading runs) |
| `sync` | `false` | load + index (baseline DDL exception) | empty |

### 5.3 Weaviate server configuration ([config/weaviate.env](config/weaviate.env), [docker-compose.yml](docker-compose.yml))

| Variable | Value | Reason |
|---|---|---|
| `PERSISTENCE_DATA_PATH` | `/db` | baseline path |
| `GOMAXPROCS` | `4` | = pinned cores; index build uses at most 4 threads |
| `GOMEMLIMIT` | `6800MiB` | ~85% of the 8 GiB limit (Weaviate docs: 80–90%); GC works harder before the OOM killer |
| `LIMIT_RESOURCES` | `false` | when true it overrides `GOMEMLIMIT` and uses cores − 1 |
| `ASYNC_INDEXING` | `true` / `false` | timing mode, see 5.2 |
| `DEFAULT_VECTORIZER_MODULE`, `ENABLE_MODULES` | `none`, empty | only self-provided embeddings |
| `DEFAULT_QUANTIZATION` | `none` | compression only when a profile asks for it |
| `AUTOSCHEMA_ENABLED` | `false` | no implicit schema changes |
| `QUERY_MAXIMUM_RESULTS` | `10000` | k = 100 with headroom |
| `PERSISTENCE_HNSW_MAX_LOG_SIZE` | `1GiB` | docs: close to the graph size (0.1–1.3 GB here) |
| `PROMETHEUS_MONITORING_ENABLED` | `true` | RSS sampling during load/index only |
| `DISABLE_TELEMETRY`, `LOG_LEVEL=warning`, anonymous access | | less noise |

### 5.4 Index configuration: profiles chosen by dataset size ([config/profiles.toml](config/profiles.toml))

The index type and parameters are free for each group, and the right values
depend on the dataset size, so the configuration is chosen **automatically from
the number of rows and the dimension**, and **every value can be overridden by
hand**.

1. **Selection rules** (first match wins) on `raw_gb = rows × dim × 4 / 1e9`
   (also `min/max_dim`, `min/max_rows`):

   | Profile | Rule | Index | `maxConnections` | `efConstruction` | Datasets |
   |---|---|---|---|---|---|
   | `small` | raw ≤ 1.5 GB | HNSW | 48 | 256 | glove-200 |
   | `medium` | raw ≤ 3 GB | HNSW | 32 | 192 | celeba-2048, landmark-768 |
   | `large` | raw ≤ 5.34 GB | HNSW | 24 | 128 | arxiv-768, inaturalist-2048 |

   Smaller datasets get a denser graph (higher recall at the same `ef`), larger ones
   a sparser graph to stay within memory and keep the 4-core build time reasonable.
   These are starting values to be tuned empirically.

2. **Memory estimator** decides whether a configuration fits the container:
   - `simple`: `vector_bytes × 1.5 ≤ 8 GB`. With no quantization and a full cache this is exactly the baseline rule *raw × 1.5 ≤ container RAM*.
   - `detailed` (default): `(vector_bytes + rows × 2 × maxConnections × bytes_per_link) × 1.25 ≤ 8 GB × 0.85`. Here `bytes_per_link` is 10 B by default (Weaviate docs: ~8–10 B per connection).
   - `vector_bytes` accounts for quantization and for `vector_cache_max_objects`.

3. **Oversize policy.** When no rule matches or the estimate does not fit:
   - `reject` (default): the run stops. hotpotqa-640 (13.4 GB) is rejected, as in the baseline.
   - `fallback`: a profile from the table below is used instead.

   | Fallback profile | What it does |
   |---|---|
   | `hnsw_rq8`, `hnsw_rq4` | HNSW + rotational quantization (4× / 8× smaller vectors in RAM, full vectors on disk) |
   | `hnsw_pq`, `hnsw_sq`, `hnsw_bq` | HNSW + product / scalar / binary quantization |
   | `hfresh` | Weaviate's cluster-based, disk-oriented index (SPFresh, RQ 1-bit) |
   | `flat` | no index, exact search (the baseline's "no index" reference); can also be chosen manually |

4. **Defaults shared by all profiles** (all overridable):
   - `distance = auto`: taken from the file, and "normalized"/"any" → cosine.
   - `quantizer = none`.
   - `vector_cache_max_objects = 1e12`: everything in RAM. A value below the row count keeps part of the vectors on disk.
   - `ef = -1`: dynamic. The benchmark sets explicit values for its sweep, because `ef` can be changed without a rebuild.

**Precedence** (later wins): `[defaults]` → profile (automatic, or manual via
`--profile`, `WVB_PROFILE` or `profile = "..."` in `[overrides."<dataset>"]`) →
`[overrides."<dataset>"]` → `WVB_PARAMS="k=v,k=v"` → `--param k=v`. Keys with a dot
address other sections, e.g. `--param ingest.mode=server_side`. Unknown keys are
an error. Print the resolved configuration without touching the database:

```bash
docker compose run --rm --no-deps benchmark python scripts/show_config.py --all
```

### 5.5 Populating (baseline §6)

`docker compose run --rm setup <dataset>` runs the steps in the baseline order. Each step also exists as its own script.

1. **Download** (`scripts/download.py`):
   - fetches the file into `/data` through a `.part` file
   - prints the size and sha256 and compares them with [config/datasets.toml](config/datasets.toml)
2. **Schema** (`scripts/ddl.py`):
   - drops the collection, restarts Weaviate so the container memory peak starts from zero, then creates the collection
3. **Load** (`scripts/load.py`):
   - reads `train` in chunks of 50,000 rows
   - converts uint8 → float32 only when needed (no other preprocessing)
   - sends the objects in one of three `ingest.mode`s:
     - `client_fixed` (default): client-side batches of `batch_size` with `concurrent_requests`
     - `client_dynamic`: the client adapts the batch size
     - `server_side`: Weaviate's server-side batching. One gRPC stream; the server tells the client how many objects to send next.
   - any failed object stops the run with a non-zero exit code
4. **Index** (`scripts/index.py`):
   - waits until the vector queue is empty on every shard
   - measures index and database size on disk
5. **Check** (`scripts/check.py`):
   - row count equals the number of train rows
   - the dimension matches
   - 10 random ids (seed 42) return the same vectors as the file
6. **Result**: one row in `results/build.csv`, plus `results/runs/<time>_build_<dataset>.json`. The JSON holds the resolved configuration, server env, versions, CPU, RAM, OS and date.

### 5.6 Measurements

| Quantity | How |
|---|---|
| ingest_s / build_s | `time.perf_counter()` in the client, see 5.2 |
| peak_mem_mb | cgroup `memory.peak` of the Weaviate container (container memory peak), read through the Docker socket. Weaviate is restarted before loading, so the peak covers load + index (and later queries). Counts the container's page cache too. Fallback: Prometheus `process_resident_memory_bytes`. |
| RSS sampling | Prometheus every 0.5 s **only during load/index**. During search it is never polled, because scraping `/metrics` next to millisecond queries would distort p99. |
| index_mb | HNSW commit logs/snapshots, compressed vectors and HFresh/flat files in the collection directory |
| db_mb | whole `/db` after loading and indexing |

## 6. Goals

`db <goal> <params...>` prints one result per line as `id distance`, nearest
first; a failure returns a non-zero exit code. The dataset comes from
`--dataset` or `WVB_DATASET`.

| # | Goal | Parameters | Example |
|---|---|---|---|
| 1 | KNN | `<query_id> <k>`: index of a test vector, number of neighbours | `db 1 0 10` |
| 2 | K-means | parameters agreed in class | TBD |

**Goal 1 — KNN.**
- Reads `test[query_id]` from the dataset file.
- Runs a `nearVector` query on the named vector with `limit = k`, returning no properties, only the distance.
- Decodes the id from the object UUID.
- With HNSW the search width is `ef` (dynamic by default).

```bash
docker compose run --rm benchmark db 1 0 10
```

## 7. Team roles

| Name | Role | Responsibilities |
|---|---|---|
| TBD | TBD | TBD |
| TBD | TBD | TBD |

## 8. Results

`results/build.csv` and `results/search.csv` use the baseline columns:

- `build.csv`: group, system, version, dataset, ingest_s, build_s, index_mb, db_mb, peak_mem_mb, cpu, cores, ram_gb, date, kmeans_s
- `search.csv`: group, system, version, dataset, k, setting_name, setting_value, recall, p50_ms, p95_ms, p99_ms, qps, peak_mem_mb (milestone 2)

*To be filled after the runs.*

## 9. User manual

### Commands

```bash
docker compose build                                    # client image
docker compose up -d --wait weaviate                    # database
docker compose run --rm download glove-200-cosine       # 1. download (size + sha256)
docker compose run --rm setup glove-200-cosine          # 2-5. schema, load, index, check -> build.csv
docker compose run --rm benchmark db 1 0 10             # goal 1 (WVB_DATASET from .env)
docker compose run --rm benchmark db 1 0 10 --dataset glove-200-cosine
```

Single steps and options:

```bash
docker compose run --rm benchmark python scripts/ddl.py   <dataset>
docker compose run --rm benchmark python scripts/load.py  <dataset>
docker compose run --rm benchmark python scripts/index.py <dataset>
docker compose run --rm benchmark python scripts/check.py <dataset>
docker compose run --rm --no-deps benchmark python scripts/show_config.py <dataset> [--json]

# options of setup / ddl / load / index / check / show_config
--profile medium                       # manual profile
--param max_connections=32             # any index parameter
--param ingest.mode=server_side        # any other setting
--param oversize.policy=fallback       # allow a fallback for oversized datasets
--allow-oversize                       # run although the estimate does not fit
```

The same through `make`: `make up`, `make download DS=…`, `make setup DS=… ARGS="--param …"`,
`make db Q="1 0 10"`, `make show-config DS=--all`, `make mem-peak`, `make test`, `make clean`.

### Reproducing the results

1. Docker Desktop: VM with ≥ 5 CPUs and ~10 GB RAM; laptop plugged in, heavy apps closed.
2. `cp .env.example .env` and keep the default limits. `ASYNC_INDEXING=true` is the default timing mode.
3. `docker compose build && docker compose up -d --wait weaviate`
4. For every dataset in the table of 5.4:
   - `docker compose run --rm download <dataset>`
   - `docker compose run --rm setup <dataset>`
5. Results are in `results/build.csv`; the full configuration of each run is in `results/runs/`.

Datasets: `glove-200-cosine`, `celeba-resnet-2048-cosine`, `landmark-nomic-768-normalized`,
`arxiv-nomic-768-normalized`, `inaturalist-resnet-2048-cosine`
(`hotpotqa-harrier-640-normalized` is rejected by the baseline memory rule).

## 10. Self-evaluation

*To be filled after the performance analysis.* Known limitations so far:
- Weaviate cannot build the index strictly after loading. The `async` mode separates the two times, but they overlap.
- The cgroup memory peak includes page cache.
