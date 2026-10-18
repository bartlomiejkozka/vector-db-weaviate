"""KNN (K-Nearest Neighbors) query implementation.

Usage: db 1 <vector_id> <k>

Finds the K nearest neighbors of the vector with the given ID
in the Landmark collection using Weaviate's nearVector search.
"""

import sys
import numpy as np
from pathlib import Path

from src.client import get_client

COLLECTION_NAME = "Landmark"
GROUND_TRUTH_PATH = Path("/app/data/ground_truth.npz")


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
    
    # Load ground truth to get the query vector
    if not GROUND_TRUTH_PATH.exists():
        print("Error: Ground truth file not found. Run data loading first.", file=sys.stderr)
        sys.exit(1)
    
    gt = np.load(GROUND_TRUTH_PATH)
    test_vectors = gt['test_vectors']
    gt_neighbors = gt['neighbors']
    gt_distances = gt['distances']
    
    if vector_id >= len(test_vectors):
        print(f"Error: vector_id {vector_id} out of range (max: {len(test_vectors) - 1})", file=sys.stderr)
        sys.exit(1)
    
    query_vector = test_vectors[vector_id].tolist()
    
    # Query Weaviate
    client = get_client()
    try:
        collection = client.collections.get(COLLECTION_NAME)
        
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=k,
            return_metadata=["distance"],
            return_properties=["vector_id", "split"],
        )
        
        print(f"KNN Results for test vector #{vector_id} (k={k}):")
        print()
        
        result_ids = []
        for i, obj in enumerate(response.objects):
            vid = obj.properties.get("vector_id")
            dist = obj.metadata.distance
            split = obj.properties.get("split")
            result_ids.append(vid)
            print(f"  {i + 1}. vector_id={vid} (split={split}) distance={dist:.6f}")
        
        # Compare with ground truth
        if vector_id < len(gt_neighbors):
            gt_nn = gt_neighbors[vector_id][:k]
            gt_dist = gt_distances[vector_id][:k]
            
            # Calculate recall
            result_set = set(result_ids)
            gt_set = set(gt_nn.tolist())
            recall = len(result_set & gt_set) / len(gt_set) if len(gt_set) > 0 else 0
            
            print(f"\nGround truth (top {k}):")
            for i, (nn_id, nn_dist) in enumerate(zip(gt_nn, gt_dist)):
                print(f"  {i + 1}. vector_id={nn_id} distance={nn_dist:.6f}")
            print(f"\nRecall@{k}: {recall:.4f} ({int(recall * k)}/{k} correct)")
    
    finally:
        client.close()
