"""Download and load the landmark-nomic-768-normalized dataset into Weaviate."""

import os
import sys
import time
import h5py
import numpy as np
import requests
import weaviate.classes.config
from pathlib import Path

from src.client import get_client

DATASET_URL = "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/landmark-nomic-768-normalized.hdf5"
DATA_DIR = Path("/app/data")
DATASET_PATH = DATA_DIR / "landmark-nomic-768-normalized.hdf5"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.npz"
COLLECTION_NAME = "Landmark"


def download_dataset():
    """Download the dataset if not already present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if DATASET_PATH.exists():
        print(f"Dataset already exists at {DATASET_PATH}, skipping download.")
        return
    
    print(f"Downloading dataset from {DATASET_URL}...")
    response = requests.get(DATASET_URL, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    chunk_size = 8192 * 1024  # 8MB chunks
    
    with open(DATASET_PATH, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = downloaded / total_size * 100
                print(f"\rDownloading: {downloaded / 1e9:.2f} / {total_size / 1e9:.2f} GB ({pct:.1f}%)", end="", flush=True)
    
    print(f"\nDownload complete: {DATASET_PATH}")


def load_data():
    """Load the dataset into Weaviate."""
    print("Opening HDF5 file...")
    with h5py.File(DATASET_PATH, 'r') as f:
        train_vectors = f['train'][:]
        test_vectors = f['test'][:]
        neighbors = f['neighbors'][:]
        distances = f['distances'][:]
    
    print(f"Train vectors: {train_vectors.shape}")
    print(f"Test vectors:  {test_vectors.shape}")
    print(f"Neighbors:     {neighbors.shape}")
    print(f"Distances:     {distances.shape}")
    
    # Save ground truth for query verification
    np.savez(GROUND_TRUTH_PATH, neighbors=neighbors, distances=distances, test_vectors=test_vectors)
    print(f"Ground truth saved to {GROUND_TRUTH_PATH}")
    
    # Connect to Weaviate
    client = get_client()
    
    try:
        # Delete existing collection if it exists
        if client.collections.exists(COLLECTION_NAME):
            print(f"Deleting existing collection '{COLLECTION_NAME}'...")
            client.collections.delete(COLLECTION_NAME)
        
        # Create collection
        from weaviate.classes.config import Configure, Property, DataType
        
        collection = client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=weaviate.classes.config.VectorDistances.COSINE,
            ),
            properties=[
                Property(name="vector_id", data_type=DataType.INT),
                Property(name="split", data_type=DataType.TEXT),
            ],
        )
        print(f"Created collection '{COLLECTION_NAME}'")
        
        # Import train vectors
        print(f"Importing {len(train_vectors)} train vectors...")
        start_time = time.time()

        # Client-side batching!!! (consider change to Server-side batching as more efficient and stable?)
        with collection.batch.fixed_size(batch_size=1000) as batch: 
            for i, vector in enumerate(train_vectors):
                batch.add_object(
                    properties={"vector_id": i, "split": "train"},
                    vector=vector.tolist(),
                )
                if (i + 1) % 10000 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    print(f"  Imported {i + 1}/{len(train_vectors)} train vectors ({rate:.0f} vec/s)")
        
        train_time = time.time() - start_time
        print(f"Train vectors imported in {train_time:.1f}s")
        
        # Import test vectors
        print(f"Importing {len(test_vectors)} test vectors...")
        test_start = time.time()

        # Client-side batching!!! (consider change to Server-side batching as more efficient and stable?)
        with collection.batch.fixed_size(batch_size=1000) as batch:
            for i, vector in enumerate(test_vectors):
                batch.add_object(
                    properties={"vector_id": len(train_vectors) + i, "split": "test"},
                    vector=vector.tolist(),
                )
                if (i + 1) % 5000 == 0:
                    elapsed = time.time() - test_start
                    rate = (i + 1) / elapsed
                    print(f"  Imported {i + 1}/{len(test_vectors)} test vectors ({rate:.0f} vec/s)")
        
        test_time = time.time() - test_start
        total_time = time.time() - start_time
        
        print(f"\n=== Import Summary ===")
        print(f"Train vectors: {len(train_vectors)} in {train_time:.1f}s")
        print(f"Test vectors:  {len(test_vectors)} in {test_time:.1f}s")
        print(f"Total time:    {total_time:.1f}s")
        print(f"Total objects:  {len(train_vectors) + len(test_vectors)}")
        
    finally:
        client.close()


def main():
    download_dataset()
    load_data()


if __name__ == "__main__":
    main()
