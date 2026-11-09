# Plan: Weaviate environment for the Common Baseline (Milestone 1)

## Context

The repo contains a rough prototype (`src/`, `config.toml`, `docker-compose.yml`, `Makefile`, `db`) that is going to be replaced. The goal of milestone 1 is a ready, reproducible environment: a Weaviate container plus a client container, VIBE dataset download, schema/DDL, loading, index, a correctness check and the first goal `db 1` (KNN). That leaves only measuring for milestone 2 (an `ef` sweep → `search.csv`, clustering).

The common requirements (Common Baseline, PDF) are fixed: datasets and rules, 4 pinned cores, 8 GB RAM with swap off, the paths `/data` `/db` `/results`, the order of steps, the CLI `db <goal> <params>`, the `build.csv`/`search.csv` format, pinned versions.

What we can influence is the **Weaviate configuration**: the server environment variables, and the index type and its parameters. The parameters are chosen automatically from the dataset size (N, d), with the option to override them by hand. The Gemini file (`specyfikacja_konfiguracji_weaviate_pod_common_baseline.md`) is used only as a side reference: it targets the outdated version 1.27, uses the legacy API and contains mistakes, e.g. collection names without a capital first letter.

User decisions:
- The timing mode is chosen in the config: `async` (`ingest_s` and `build_s` separately) or `sync` (the DDL exception: a single `ingest_s`, empty `build_s`).
- HNSW only by default. A dataset that exceeds the RAM budget is **rejected** by default. The config can select a fallback (HNSW+RQ/PQ/BQ/SQ or HFresh).
- The memory estimator has two modes: simple (`raw × 1.5`, the rule from the PDF) and detailed (vectors + HNSW edges × bytes per link).

## Target repo structure

Old files to remove: `src/`, `config.toml`, `docker-compose.yml`, `Makefile` and `db`. All of them will be rewritten.

```
Dockerfile                  # client image (python:3.12-slim, pinned dependencies)
docker-compose.yml          # weaviate + download + benchmark (+ load/index/check)
.env.example                # CPU pinning / ports / versions - local overrides
config/
  weaviate.env              # SERVER environment variables (static)
  datasets.toml             # VIBE dataset registry (url, sha256, N, d, distance)
  profiles.toml             # thresholds, estimator, index profiles, per-dataset overrides
scripts/                    # thin wrappers required by the baseline
  download.py  ddl.py  load.py  index.py  check.py  show_config.py  benchmark.py (M2 stub)
src/wvbench/
  settings.py   # loading TOML + env + CLI, precedence resolution
  sizing.py     # memory estimator + profile selection (auto)
  dataset.py    # HDF5 access (train/test/neighbors, point_type, distance attr), chunking
  client.py     # connection (gRPC), wait-for-ready
  schema.py     # DDL: SCHEMA part / INDEX part
  ingest.py     # bulk load, 50k per chunk, batches to Weaviate
  indexing.py   # waiting for vectorQueueLength==0, build_s measurement
  metrics.py    # timers, peak RSS sampler (Prometheus), /db size, sysinfo
  results.py    # writing build.csv / search.csv + JSON log with the full configuration
  goals/knn.py  # goal 1
  cli.py        # `db <goal> <params...>`
db                          # CLI entry point in the container (exec python -m wvbench.cli)
results/                    # build.csv, search.csv, runs/*.json (committed)
README.md
```

## 1. Deployment (docker-compose + Dockerfile)

- **`weaviate`**: `cr.weaviate.io/semitechnologies/weaviate:1.40.1` (pinned). Settings:
  - `cpuset: ${WEAVIATE_CPUS:-0-3}`, `mem_limit: 8g`, `memswap_limit: 8g`
  - `env_file: config/weaviate.env`
  - named volume `weaviate_db:/db`
  - healthcheck `/v1/.well-known/ready`
  - ports 8080, 50051 and 2112 (metrics) published to the host only optionally
- **Client image** (`Dockerfile`): `python:3.12-slim` with `requirements.txt` pinned via `==` (weaviate-client, h5py, numpy, requests). The versions are frozen during implementation.
- **`download`**: `/data` mounted read-write, runs `scripts/download.py <dataset>`.
- **`benchmark`**: `/data:ro`, `weaviate_db:/db:ro` (for size measurement), `./results:/results`. Settings `cpuset: ${CLIENT_CPUS:-4}`, `depends_on: weaviate (healthy)`, entrypoint `db`. The `load`, `index` and `check` commands are the same image with a different entrypoint (`scripts/*.py`).
- Commands compatible with the PDF:
  - `docker compose build`
  - `docker compose run --rm download <ds>`
  - `docker compose run --rm benchmark db 1 443321 4`
- The README has to note that on macOS the Docker VM needs ≥5 vCPU and ~10 GB RAM. `WEAVIATE_CPUS` and `CLIENT_CPUS` can be overridden in `.env`.

## 2. Server environment variables - `config/weaviate.env`

These values are static, because none of them depends on the dataset. Everything that depends on the dataset is a collection setting (section 3).

| Variable | Value | Why |
|---|---|---|
| `PERSISTENCE_DATA_PATH` | `/db` | baseline path |
| `GOMAXPROCS` | `4` | = pinned cores (the index is built with ≤4 threads) |
| `GOMEMLIMIT` | `6800MiB` | ~85% of 8 GiB, the docs recommend 80-90%; the GC kicks in before the OOM killer |
| `LIMIT_RESOURCES` | `false` | otherwise it overrides GOMEMLIMIT and takes cores-1 |
| `ASYNC_INDEXING` | `true` | needed in `async` mode; in `sync` mode compose gets `false` through `.env` (section 5) |
| `DEFAULT_VECTORIZER_MODULE` / `ENABLE_MODULES` | `none` / `` | pure embeddings, no modules |
| `DEFAULT_QUANTIZATION` | `none` | compression only explicitly, from a profile |
| `QUERY_MAXIMUM_RESULTS` | `10000` | k=100 with headroom |
| `PERSISTENCE_HNSW_MAX_LOG_SIZE` | `1GiB` | the docs recommend a value close to the graph size (~0.3-0.7 GB for us) |
| `PROMETHEUS_MONITORING_ENABLED` | `true` | auxiliary view of the Weaviate process RSS during load/index; never polled during search (section 5) |
| `DISABLE_TELEMETRY`, `LOG_LEVEL=warning`, `AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true`, `CLUSTER_HOSTNAME=node1` | | noise reduction |

## 3. Index configuration: `config/profiles.toml` + `sizing.py`

### Dataset registry (`datasets.toml`)

For every dataset from the PDF (glove-200, landmark-768, arxiv-768, celeba-2048, inaturalist-2048, hotpotqa-640) the registry stores:
- `url` (HF `vector-index-bench/vibe`)
- the expected `rows`, `dim`, `distance`
- `sha256`, filled in after the first download

The real N, d, `point_type` and `distance` are always read from the HDF5 attributes. The registry is used for validation only.

### Memory estimator (`[estimator]`)

```toml
[estimator]
mode = "detailed"            # "simple" | "detailed"
memory_limit_gb = 8.0
simple_factor = 1.5          # baseline rule: raw*1.5 <= limit
bytes_per_link = 10          # Weaviate docs: ~8-10 B/connection (configurable up to 20)
overhead_factor = 1.25       # GC, copies, allocator
budget_fraction = 0.85       # compared with GOMEMLIMIT, not with 8 GB
```

- **simple**: `N·d·4 × 1.5 ≤ memory_limit`
- **detailed**: `(N·d·4 / compression + N · 2·maxConnections · bytes_per_link) × overhead ≤ memory_limit × budget_fraction`
  - HNSW layer 0 has 2·M connections. `compression` comes from the quantizer (none=1, sq/rq8=4, rq4=8, bq/rq1=32, pq=4·d/segments).
- The estimator result, i.e. the estimate in MB, the rule used and the decision, goes into the run log.

### Profile selection (rules, the first match wins)

The criteria are `max_raw_gb`, `min_dim` and `max_dim`, so both N and d matter. The starting values below are to be tuned empirically in M2:

| Profile | Condition | index | M (maxConnections) | efConstruction | ef sweep (M2) |
|---|---|---|---|---|---|
| `small` | raw ≤ 1.5 GB | hnsw | 48 | 256 | 16…512 |
| `medium` | raw ≤ 3 GB | hnsw | 32 | 192 | 16…512 |
| `large` | raw ≤ 5.34 GB | hnsw | 24 | 128 | 16…512 |
| `oversize` | the estimator exceeds the budget | per `oversize.policy` | | | |

```toml
[oversize]
policy = "reject"            # "reject" (default) | "fallback"
fallback = "hnsw_rq8"        # profile name: hnsw_rq8 | hnsw_rq4 | hnsw_pq | hnsw_bq | hnsw_sq | hfresh
```

The fallback profiles are defined in the same file:
- `hnsw_rq8`: `quantizer={type="rq", bits=8, rescore_limit=20}`
- `hnsw_pq`: `segments=d/8`, `training_limit=100000`
- `hfresh`: RQ 1-bit forced by Weaviate, `max_posting_size_kb=48`, `replicas=4`; the sweep parameter is `search_probe`

After a fallback is chosen, the estimator is computed again with the compression. If it still does not fit, the run fails. `--allow-oversize` bypasses this check.

There is also a `flat` profile (no index, brute force). The PDF lists it as a baseline, and it is useful if the instructor forbids indexes held in RAM. It can be chosen manually with `--profile flat`.

**The `[defaults]` values** are inherited by every profile. All of them can be overridden per profile, per dataset (`[overrides.<ds>]`), through `WVB_*` or `--param`; nothing is hard-coded.
- `distance = "auto"`: from the file, and "any" → cosine
- `quantizer = "none"`
- `vector_cache_max_objects = 1e12`: the whole dataset in RAM. A smaller value, e.g. 0.3·N, forces part of the vectors to be read from disk. This is the "not everything in RAM" scenario, should the instructor ask for it.
- `flat_search_cutoff = 40000`: irrelevant, because we have no filters
- `ef = -1`: dynamic ef computed by Weaviate from `dynamicEfMin/Max/Factor` and the k limit. `ef` is mutable, so in the benchmark (M2) before every sweep point we run `collection.config.update(... hnsw(ef=X))`, which overrides -1. After the sweep we restore -1. During implementation we have to check whether Weaviate uses `max(ef, k)` for k=100; if so, the sweep for k=100 starts at 100.

### Precedence (highest first)

`--param key=value` (CLI) > `WVB_*` variables > `[overrides.<dataset>]` in `profiles.toml` > `--profile <name>` > automatic selection.

`scripts/show_config.py <dataset>` prints the resolved configuration, the estimate and the reason for the choice without touching the database.

## 4. Schema / DDL - `src/wvbench/schema.py` (+ `scripts/ddl.py`)

- The collection name is the file name without the extension, with `-`→`_`. **Weaviate enforces a capital first letter**, so `glove_200_cosine` → `Glove_200_cosine`. The mapping is described in the README.
- The baseline `id` column: `id` is reserved in Weaviate (it is the UUID), so:
  - object UUID = `uuid.UUID(int=row)`. The mapping is reversible and gives "ids only" without reading any property.
  - the property `row_id: INT` (int64, `index_filterable=True`, `index_searchable=False`) is the explicit required column. It is controlled by `[schema] store_row_id = true` (the default).
    - `false` means UUID only: no property, and the id is computed from the UUID.
    - With `true` the cost is small: ~8 B per object plus a filterable bitmap, i.e. tens of MB. It slightly lengthens ingest and satisfies the "id column" requirement literally.
    - Queries, the check and goal 1 **always** use the UUID (`return_properties=[]`), so `row_id` does not affect latency.
- The `vector` column: a named vector `Configure.Vectors.self_provided(name="vector", vector_index_config=...)`. If the name "vector" turns out to be reserved, we use `"embedding"` and record it in the README.
- `sharding desired_count=1`, `replication factor=1`.
- `scripts/ddl.py` has two marked parts: `# === SCHEMA PART ===` (drop + create, idempotent) and `# === INDEX PART ===` (index parameters from the profile). Because in Weaviate the index is part of the collection definition, the INDEX PART is applied at create time, and the README explains this (section 5). The README also gets a table of the columns (`row_id` int64 / `vector` float32[d]).

## 5. Pipeline: download → schema → load → index → check

1. **download.py**: streams into `/data/<name>.hdf5.part` and renames it after the download. Prints the size and sha256 and compares them with `datasets.toml`. An existing file is only verified.
2. **ddl.py (schema)**: drop and create according to the resolved profile.
3. **load.py**:
   - reads `train` in chunks of 50,000
   - converts `uint8`→`float32` if needed; otherwise changes nothing
   - the sending mode is chosen in `profiles.toml [ingest] mode`:
     - `client_fixed` (default, as in the baseline): `batch.fixed_size(batch_size, concurrent_requests)`
     - `client_dynamic`: `batch.dynamic()`, the client picks the batch size based on server load
     - `server_side`: server-side batching. The client opens a stream and the server itself says how many objects it wants (backpressure). The exact API will be checked during implementation against the pinned client and server versions (e.g. `client.batch.experimental()` / `stream()`). If the version does not support it, the run ends with a clear error.
     - In all modes the file is read in chunks of 50,000, and the mode goes into the run log.
   - after the last chunk it checks `failed_objects` and exits non-zero on errors
   - measures `ingest_s` (perf_counter) and peak RSS
4. **index.py**, depending on `timing.mode` in `profiles.toml`:
   - `async` (ASYNC_INDEXING=true): `build_s` = from the end of the load until the sum of `vectorQueueLength` == 0 and the shard status ≠ INDEXING (`client.cluster.nodes(output="verbose")`, polled every 1 s). The README notes that graph building already starts during loading.
   - `sync` (ASYNC_INDEXING=false): the DDL exception, `ingest_s` = load+index, `build_s` empty.
   - Consistency is enforced by `settings.py`: the mode from the config has to match the `ASYNC_INDEXING` reported by the server (`/v1/meta` or an env check). On a mismatch we get an error with instructions for switching `.env`.
5. **check.py**:
   - `count == N` (`aggregate.over_all`)
   - vector dimension == d
   - 10 random ids (seed 42) through `fetch_object_by_id(include_vector=True)` must give `np.allclose` with the file
   - non-zero exit on error
6. At the end we write a row to `results/build.csv` (`kmeans_s` empty) and `results/runs/<ts>_<dataset>.json` with the full configuration: profile, env of the server, versions, CPU, RAM, OS, date. The PDF requires logging this.

**Auxiliary measurements (`metrics.py`)**:
- **peak memory, main source:** the cgroup of the Weaviate container.
  - We read `/sys/fs/cgroup/memory.peak` (cgroup v2), with a fallback to `memory.max_usage_in_bytes` on cgroup v1. This is the "container memory peak" from the PDF.
  - The Weaviate container is restarted before `load`, so the peak starts from zero. Because the peak is monotonic, readings after load, after index and after search give the peak for the successive phases.
  - We read it through `docker exec weaviate cat …`: from the host (Makefile/`scripts/mem_peak.sh`) or from the client through the Docker SDK with `/var/run/docker.sock` mounted. The source is set by `[metrics] memory_source = "cgroup" | "prometheus" | "off"`.
- **Prometheus (auxiliary):** a sampler every 0.5 s (`process_resident_memory_bytes`) runs **only during load/index**, where its overhead is negligible. During search it is **disabled**, so that it does not distort p99, and we only read it before and after.
- `db_mb`: `du` on `/db` (ro)
- `index_mb`: the sum of the HNSW files (`*/main.hnsw.commitlog.d` + snapshots) in the collection directory

## 6. CLI - `db <goal> <params...>`

- `db 1 <query_id> <k>`: KNN for the vector `test[query_id]`. On stdout one line per result in the format `id distance`, nearest first, with `return_properties=[]`, and the `id` computed from the UUID. Any errors go to stderr with exit≠0.
- Additional flags (`--dataset`, `--ef`) are optional. The default dataset comes from `WVB_DATASET`.
- The dispatcher is ready for further goals (clustering, M2).

## 7. Makefile + README

- The `Makefile` is a thin wrapper over `docker compose`: `build`, `up`, `download DS=…`, `load DS=…`, `index`, `check`, `show-config DS=…`, `db Q="1 0 10"`, `mem-peak`, `clean`.
- The README covers: the checklist from the assignment, step-by-step commands, the column table, the DDL and timing modes, the table of server variables, the profiles and the estimator, and how to override parameters.

## Openness to changes after the class discussion

The uncertain things are RAM vs disk, dropping the indexes, and fewer metrics. Each of them is a config change, not a code change:
- "not everything in RAM": `vector_cache_max_objects` < N, or the profiles `hnsw_rq*`, `hnsw_pq` or `hfresh` (compressed in RAM, full vectors on disk). Weaviate has no DiskANN, and HFresh is its disk-oriented counterpart.
- "no index": the `flat` profile
- different limits: `memory_limit_gb` in `[estimator]` and `mem_limit` in `.env`
- fewer metrics: CSV columns that we do not measure stay empty

## End-to-end verification

1. `docker compose build && docker compose up -d weaviate`. We check readiness, and `docker inspect` shows cpuset 0-3, Memory 8 GiB and MemorySwap 8 GiB.
2. `scripts/show_config.py` for all 6 datasets (no database):
   - glove → small, landmark → medium, arxiv and inaturalist → large
   - hotpotqa → reject in both estimator modes
   - with `policy=fallback` hotpotqa → hnsw_rq8 or hfresh with a recomputed budget
   - unit tests of `sizing.py` (pytest) for these cases
3. `docker compose run --rm download glove-200-cosine` prints the size and sha256.
4. The full pipeline on glove (the smallest) in `async` mode, then once in `sync`. The `build.csv` row has the expected empty and filled fields, and `check` passes.
5. `docker compose run --rm benchmark db 1 0 10` gives 10 lines of `id distance`. We compare with `neighbors[0][:10]` (recall close to 1 at the default ef). `db 1 999999999 10` returns exit≠0.
6. Finally arxiv (the largest accepted): peak RSS < 8 GB, no OOM kill (`docker inspect … OOMKilled=false`).
