"""KNN (K-Nearest Neighbors) query implementation.

Usage: db 1 <vector_id> <k>

Finds the K nearest neighbors of the given test vector in the configured
collection using Weaviate's nearVector search, and compares the result
with the ground truth stored in the dataset file.
"""

import sys

import h5py

from src.client import get_client
from src.config import COLLECTION_NAME, DATASET_PATH


def run(params: list[str]):
    """Execute KNN query.

    Args:
        params: [vector_id, k]
            vector_id: ID of the query vector (index in test set)
            k: Number of nearest neighbors to find
    """
    if len(params) < 2:
        print("Usage: db 1 <vector_id> <k>", file=sys.stderr)
        print("  vector_id: Index of the test vector (0-based)", file=sys.stderr)
        print("  k: Number of nearest neighbors to find", file=sys.stderr)
        sys.exit(1)

    vector_id = int(params[0])
    k = int(params[1])

    if not DATASET_PATH.exists():
        print(f"Error: Dataset file {DATASET_PATH} not found. Run 'make load' first.", file=sys.stderr)
        sys.exit(1)

    # Read only the requested query vector and its ground truth from the dataset file
    with h5py.File(DATASET_PATH, "r") as f:
        n_test = f["test"].shape[0]
        if not 0 <= vector_id < n_test:
            print(f"Error: vector_id {vector_id} out of range (0-{n_test - 1})", file=sys.stderr)
            sys.exit(1)
        query_vector = f["test"][vector_id].tolist()
        gt_nn = f["neighbors"][vector_id][:k]
        gt_dist = f["distances"][vector_id][:k]

    client = get_client()
    try:
        collection = client.collections.get(COLLECTION_NAME)

        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=k,
            return_metadata=["distance"],
            return_properties=["vector_id"],
        )

        print(f"KNN Results for test vector #{vector_id} (k={k}):")
        print()

        result_ids = []
        for i, obj in enumerate(response.objects):
            vid = obj.properties.get("vector_id")
            result_ids.append(vid)
            print(f"  {i + 1}. vector_id={vid} distance={obj.metadata.distance:.6f}")

        gt_set = set(gt_nn.tolist())
        recall = len(set(result_ids) & gt_set) / len(gt_set) if gt_set else 0

        print(f"\nGround truth (top {k}):")
        for i, (nn_id, nn_dist) in enumerate(zip(gt_nn, gt_dist)):
            print(f"  {i + 1}. vector_id={nn_id} distance={nn_dist:.6f}")
        print(f"\nRecall@{k}: {recall:.4f} ({len(set(result_ids) & gt_set)}/{len(gt_set)} correct)")

    finally:
        client.close()
