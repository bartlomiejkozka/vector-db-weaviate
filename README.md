# Advanced Database Systems — Vector Database Project (Weaviate)

## 1. Choice of Technology

- **Database**: [Weaviate](https://weaviate.io/) v1.40.1 — open-source vector database (HNSW index), run in Docker
- **Language**: Python 3.12 (runs locally in a virtualenv)
- **Dataset**: [arxiv-nomic-768-normalized](https://huggingface.co/datasets/vector-index-bench/vibe) from the VIBE benchmark
  - 768 dimensions, normalized vectors (cosine similarity)
  - Nomic Embed text embeddings of arXiv papers
- **Deployment**: Docker Compose (Weaviate only); the application talks to it over `localhost`

## 2. Architecture

```
┌──────────────────────────┐          ┌───────────────────────────┐
│  Host (local Python)     │          │  Docker Compose           │
│                          │   REST   │                           │
│  - ./db  (CLI queries)   │─────────▶│  weaviate                 │
│  - src.load_data         │ :8080    │  (HNSW index)             │
│                          │   gRPC   │                           │
│  data/*.hdf5 (dataset)   │─────────▶│  Volume: weaviate_data    │
│                          │ :50051   │                           │
└──────────────────────────┘          └───────────────────────────┘
```

- The dataset is downloaded to `./data/` on the host and read from there (loader and CLI).
- Communication with Weaviate: REST (port 8080) + gRPC (port 50051), both published by Docker on `localhost`.
- All settings (Weaviate host/ports, dataset URL, collection name, batch sizes) live in [config.toml](config.toml).
  Weaviate host/ports can be overridden with `WEAVIATE_HOST`, `WEAVIATE_HTTP_PORT`, `WEAVIATE_GRPC_PORT`.

## 3. Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (with Docker Compose v2)
- Python 3.12 (with `venv`)
- `make` and `curl`
- Disk space for the dataset (see the download size reported by `make load`)
- Internet connection (for the initial dataset download)

## 4. Installation and Setup

```bash
git clone <repo-url>
cd <repo>

# Create .venv, install dependencies, start Weaviate in Docker
make setup

# Download and load the dataset into Weaviate
make load
```

Alternatively, without `make`:
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
docker compose up -d
.venv/bin/python -m src.load_data
```

## 5. Design and Implementation

### Data Pipeline
1. `src/load_data.py` downloads the HDF5 dataset from HuggingFace into `./data/`
2. The `train` vectors are read from the file in chunks (low memory use) and batch-imported into Weaviate with HNSW indexing
3. The `test` vectors are **not** imported: they are the queries. Ground truth (`neighbors`, `distances`) is read directly from the HDF5 file when a query runs, to compute recall

### Configuration
[config.toml](config.toml) holds all tunable values; `src/config.py` loads it (TOML via the standard-library `tomllib`).

### Collection Schema
- **Collection**: `Arxiv` (name from `config.toml`)
- **Vectorizer**: none (pre-computed embeddings)
- **Index**: HNSW with cosine distance
- **Properties**: `vector_id` (int) — index of the vector in the dataset's `train` array

### CLI Tool
The `db` script dispatches queries by number:
```
db <query_number> [param1] [param2] ...
```

## 6. Goals / Queries

| # | Query | Parameters | Description |
|---|-------|-----------|-------------|
| 1 | KNN | `<vector_id> <k>` | Find K nearest neighbors of a test vector |
| 2 | K-Means | TBD | TBD |

### Query 1: KNN
Finds the K nearest neighbors (among the train vectors) of a given test vector using Weaviate's `nearVector` search.
Compares results with ground truth and reports recall@K.

```bash
make query Q="1 100 5"    # KNN: test vector #100, k=5
# or directly:
./db 1 100 5
```

## 7. Team Roles

| Name | Role | Responsibilities |
|------|------|------------------|
| TBD | TBD | TBD |
| TBD | TBD | TBD |

## 8. Results

*To be filled after running experiments.*

## 9. User Manual

### Quick Start
```bash
make setup     # Create venv, install dependencies, start Weaviate
make load      # Download and import the dataset
make query Q="1 0 10"  # Run KNN: find 10 nearest neighbors of test vector #0
```

### All Commands
```bash
make help      # Show available commands
make setup     # Create venv, install dependencies, start Weaviate
make up        # Start Weaviate
make down      # Stop Weaviate
make load      # Download dataset and import into Weaviate
make query Q="1 100 5"  # Run a query
make status    # Show container status
make logs      # Show Weaviate logs
make clean     # Stop Weaviate, remove volume and downloaded data
make reset     # Full reset: clean + setup
```

### Reproducing Results
1. `make setup` — creates the virtualenv and starts Weaviate
2. `make load` — downloads and imports dataset
3. Run specific queries as documented in section 6

## 10. Self-Evaluation

*To be filled after performance analysis.*
