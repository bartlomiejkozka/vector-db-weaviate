"""CLI entry point for the vector database project.

Usage:
    python -m src.cli <query_number> [param1] [param2] ...
    # or via the wrapper:
    ./db <query_number> [param1] [param2] ...

Available queries:
    1  KNN  - Find K nearest neighbors of a vector
       params: <vector_id> <k>
       example: ./db 1 100 5
"""

import argparse
import sys
import time

from src.client import get_client
from src.queries import knn


def main():
    parser = argparse.ArgumentParser(
        description="Vector Database CLI - Run queries against Weaviate",
        usage="db <query_number> [params...]",
    )
    parser.add_argument(
        "query_number",
        type=int,
        help="Query number to execute (1=KNN)",
    )
    parser.add_argument(
        "params",
        nargs="*",
        help="Query-specific parameters",
    )
    
    args = parser.parse_args()
    
    # Dispatch to query handlers
    queries = {
        1: ("KNN", knn.run),
        # Future queries will be added here:
        # 2: ("K-Means", kmeans.run),
    }
    
    if args.query_number not in queries:
        print(f"Error: Unknown query number {args.query_number}", file=sys.stderr)
        print(f"Available queries:", file=sys.stderr)
        for num, (name, _) in queries.items():
            print(f"  {num}: {name}", file=sys.stderr)
        sys.exit(1)
    
    query_name, query_fn = queries[args.query_number]
    print(f"Running query #{args.query_number}: {query_name}")
    print(f"Parameters: {args.params}")
    print("-" * 40)
    
    start_time = time.time()
    query_fn(args.params)
    elapsed = time.time() - start_time
    
    print("-" * 40)
    print(f"Query completed in {elapsed:.4f}s")


if __name__ == "__main__":
    main()
