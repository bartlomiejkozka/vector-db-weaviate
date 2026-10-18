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
│  │   app        │    │   weaviate    │   │
│  │  (Python)    │───▶│  (Database)   │   │
│  │             │REST│   port 8080   │   │
│  │  - CLI (db) │gRPC│   port 50051  │   │
│  │  - Loader   │    │              │   │
│  └─────────────┘    └───────────────┘   │
│                                         │
│  Volume: weaviate_data (persistent)     │
│  Volume: app_data (dataset cache)       │
└─────────────────────────────────────────┘
```

- **app**: Python container with the CLI tool and data loader
- **weaviate**: Vector database instance with HNSW index
- Communication: REST API (port 8080) + gRPC (port 50051)

## 3. Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (with Docker Compose v2)
- `make` (optional, for convenience commands)
- ~3 GB disk space for the dataset
- Internet connection (for initial dataset download)

## 4. Installation and Setup

```bash
# Clone the repository
git clone <repo-url>
cd project

# Build and start everything
make setup

# Download and load the dataset into Weaviate
make load
```

Alternatively, without `make`:
```bash
docker compose build
docker compose up -d
docker compose exec app python -m src.load_data
```

## 5. Design and Implementation

### Data Pipeline
1. `src/load_data.py` downloads the HDF5 dataset from HuggingFace
2. Vectors are batch-imported into Weaviate with HNSW indexing
3. Ground truth (neighbors + distances) is saved locally for recall evaluation

### Collection Schema
- **Collection**: `Landmark`
- **Vectorizer**: none (pre-computed embeddings)
- **Index**: HNSW with cosine distance
- **Properties**: `vector_id` (int), `split` (text: train/test)

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
Finds the K nearest neighbors of a given test vector using Weaviate's `nearVector` search.
Compares results with ground truth and reports recall@K.

```bash
make query Q="1 100 5"    # KNN: test vector #100, k=5
# or directly:
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
make setup     # Build and start containers
make load      # Download and import dataset (~760K vectors)
make query Q="1 0 10"  # Run KNN: find 10 nearest neighbors of test vector #0
```

### All Commands
```bash
make help      # Show available commands
make setup     # Build and start all containers
make up        # Start containers
make down      # Stop containers
make load      # Download dataset and import into Weaviate
make query Q="1 100 5"  # Run a query
make status    # Show container status
make logs      # Show container logs
make clean     # Stop containers, remove volumes and data
make reset     # Full reset: clean + rebuild
```

### Reproducing Results
1. `make setup` — builds containers
2. `make load` — downloads and imports dataset
3. Run specific queries as documented in section 6

## 10. Self-Evaluation

*To be filled after performance analysis.*
