# Advanced Database Systems — Vector Database Project (Weaviate)

## 1. Choice of Technology

- **Database**: [Weaviate](https://weaviate.io/) v1.40.1 — open-source vector database
- **Language**: Python 3.12
- **Dataset**: [landmark-nomic-768-normalized](https://huggingface.co/datasets/vector-index-bench/vibe) from the VIBE benchmark
  - 760,757 corpus vectors, 768 dimensions
  - Normalized vectors (cosine similarity)
  - Nomic Embed image embeddings of landmark images
- **Deployment**: Docker Compose (fully containerized)

## 2. Architecture

The project consists of two Docker containers managed by Docker Compose:

```
┌─────────────────────────────────────────┐
│            Docker Compose               │
│                                         │
│  ┌─────────────┐    ┌───────────────┐   │
│  │   app       │    │   weaviate    │   │
│  │  (Python)   │───▶│  (Database)   │   │
│  │             │REST│   port 8080   │   │
│  │  - CLI (db) │gRPC│   port 50051  │   │
│  │  - Loader   │    │               │   │
│  └─────────────┘    └───────────────┘   │
│                                         │
│  Volume: weaviate_data (persistent)     │
│  Volume: app_data (dataset cache)       │
└─────────────────────────────────────────┘
```

- **app**: Python container containing the CLI tool and data loader
- **weaviate**: Vector database instance with HNSW index
- Communication: REST API (port 8080) + gRPC (port 50051)

## 3. Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (with Docker Compose v2)
- ~3 GB disk space for the dataset and index
- Internet connection (for initial dataset download)

## 4. Installation and Setup

Setup is completely handled by Docker without requiring any local dependencies or build tools:

```bash
# 1. Clone the repository
git clone <repo-url>
cd project

# 2. Build and start all containers in the background
docker compose up -d --build

# 3. Download the dataset and import it into Weaviate (one-time setup)
docker compose exec app python -m src.load_data

# 4. Enter the interactive CLI container
docker compose exec -it app bash
```

Once inside the container shell, `/app` is on `$PATH`, so you can immediately execute queries directly:
```bash
db 1 100 5
```

> [!TIP]
> If you prefer to stay in your host terminal without entering the container shell:
> - **Linux / macOS / Git Bash**: run `export PATH="$PWD:$PATH"` once in your terminal session, then run `db 1 100 5` (or `./db 1 100 5`).
> - **Windows (PowerShell)**: run `$env:PATH = "$PWD;$env:PATH"` once in your PowerShell session, then run `db 1 100 5` (or `./db 1 100 5`).
> - **Any OS (Docker Compose direct)**: `docker compose exec app ./db 1 100 5`

## 5. Design and Implementation

### Data Pipeline
1. `src/load_data.py` downloads the HDF5 dataset from HuggingFace directly into the shared Docker volume (`app_data`).
2. Vectors are batch-imported into Weaviate with HNSW indexing.
3. Ground truth (exact nearest neighbors + distances) is saved locally for recall evaluation.

### Collection Schema
- **Collection**: `Landmark`
- **Vectorizer**: `none` (pre-computed embeddings)
- **Index**: HNSW with cosine distance
- **Properties**: `vector_id` (int), `split` (text: train/test)

### CLI Tool (`db`)
The CLI tool allows running queries by number and accepts arbitrary parameters:
```bash
db <query_number> [param1] [param2] ...
```


## 6. Goals / Queries

| # | Query | Parameters | Description |
|---|-------|-----------|-------------|
| 1 | KNN | `<vector_id> <k>` | Find K nearest neighbors of a test vector |
| 2 | K-Means | TBD | TBD |

### Query 1: KNN
Finds the $K$ nearest neighbors of a given test vector using Weaviate's `nearVector` search.
Compares results against the dataset's precomputed ground truth and reports `recall@K`.

```bash
# Example: Find 5 nearest neighbors for test vector #100
db 1 100 5

# Or via Docker Compose directly:
docker compose exec app ./db 1 100 5
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
# 1. Build and start containers
docker compose up -d --build

# 2. Download and import dataset (~760K vectors)
docker compose exec app python -m src.load_data

# 3. Enter the CLI shell
docker compose exec -it app bash

# 4. Run queries directly
db 1 0 10
```

### Command Reference

| Task | Command |
|---|---|
| **Build & start services** | `docker compose up -d --build` |
| **Stop services** | `docker compose down` |
| **Download & load dataset** | `docker compose exec app python -m src.load_data` |
| **Run query** | `db 1 <vector_id> <k>` (or `docker compose exec app ./db 1 <vector_id> <k>`) |
| **Container status** | `docker compose ps` |
| **View logs** | `docker compose logs -f` |
| **Full clean / reset** | `docker compose down -v` |

### Reproducing Results
1. Run `docker compose up -d --build` to spin up Weaviate and the application container.
2. Run `docker compose exec app python -m src.load_data` to import the benchmark dataset and extract ground truth.
3. Run the evaluation queries as documented in Section 6 (e.g. `db 1 100 5`).

## 10. Self-Evaluation

*To be filled after performance analysis.*
