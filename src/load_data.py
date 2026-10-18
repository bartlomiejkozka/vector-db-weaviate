"""Download the configured dataset and load its train vectors into Weaviate."""

import time

import h5py
import requests
import weaviate.classes.config as wvc

from src.client import get_client
from src.config import (
    BATCH_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    DATASET_PATH,
    DATASET_URL,
    READ_CHUNK,
)


def download_dataset():
    """Download the dataset if not already present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATASET_PATH.exists():
        print(f"Dataset already exists at {DATASET_PATH}, skipping download.")
        return

    print(f"Downloading dataset from {DATASET_URL}...")
    tmp_path = DATASET_PATH.with_suffix(".part")
    response = requests.get(DATASET_URL, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 8 * 1024 * 1024  # 8MB chunks

    with open(tmp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = downloaded / total_size * 100
                print(f"\rDownloading: {downloaded / 1e9:.2f} / {total_size / 1e9:.2f} GB ({pct:.1f}%)", end="", flush=True)

    # Rename only after a complete download, so an interrupted run is not mistaken for a valid file
    tmp_path.rename(DATASET_PATH)
    print(f"\nDownload complete: {DATASET_PATH}")


def load_data():
    """Load the train vectors into Weaviate, reading the HDF5 file in chunks."""
    client = get_client()

    try:
        with h5py.File(DATASET_PATH, "r") as f:
            train = f["train"]
            total = train.shape[0]
            print(f"Train vectors: {train.shape}")
            print(f"Test vectors:  {f['test'].shape} (used as queries, not imported)")
            print(f"Neighbors:     {f['neighbors'].shape}")
            print(f"Distances:     {f['distances'].shape}")

            if client.collections.exists(COLLECTION_NAME):
                print(f"Deleting existing collection '{COLLECTION_NAME}'...")
                client.collections.delete(COLLECTION_NAME)

            collection = client.collections.create(
                name=COLLECTION_NAME,
                vectorizer_config=wvc.Configure.Vectorizer.none(),
                vector_index_config=wvc.Configure.VectorIndex.hnsw(
                    distance_metric=wvc.VectorDistances.COSINE,
                ),
                properties=[
                    wvc.Property(name="vector_id", data_type=wvc.DataType.INT),
                ],
            )
            print(f"Created collection '{COLLECTION_NAME}'")

            print(f"Importing {total} train vectors...")
            start_time = time.time()
            imported = 0

            with collection.batch.fixed_size(batch_size=BATCH_SIZE) as batch:
                for start in range(0, total, READ_CHUNK):
                    chunk = train[start:start + READ_CHUNK]
                    for offset, vector in enumerate(chunk):
                        batch.add_object(
                            properties={"vector_id": start + offset},
                            vector=vector.tolist(),
                        )
                    imported += len(chunk)
                    elapsed = time.time() - start_time
                    print(f"  Imported {imported}/{total} ({imported / elapsed:.0f} vec/s)")

            if batch.number_errors:
                print(f"WARNING: {batch.number_errors} objects failed to import")
                for err in collection.batch.failed_objects[:5]:
                    print(f"  {err.message}")

            total_time = time.time() - start_time
            print("\n=== Import Summary ===")
            print(f"Train vectors: {total} in {total_time:.1f}s")
            print(f"Objects in collection: {collection.aggregate.over_all(total_count=True).total_count}")
    finally:
        client.close()


def main():
    download_dataset()
    load_data()


if __name__ == "__main__":
    main()
