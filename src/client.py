"""Weaviate client connection helper."""

import sys
import weaviate

from src.config import WEAVIATE_GRPC_PORT, WEAVIATE_HOST, WEAVIATE_HTTP_PORT


def get_client() -> weaviate.WeaviateClient:
    """Create and return a connected Weaviate client.

    Connection parameters come from config.toml (see src/config.py).
    """
    try:
        return weaviate.connect_to_custom(
            http_host=WEAVIATE_HOST,
            http_port=WEAVIATE_HTTP_PORT,
            http_secure=False,
            grpc_host=WEAVIATE_HOST,
            grpc_port=WEAVIATE_GRPC_PORT,
            grpc_secure=False,
        )
    except Exception as e:
        print(
            f"Error: Could not connect to Weaviate at {WEAVIATE_HOST}:{WEAVIATE_HTTP_PORT}: {e}",
            file=sys.stderr,
        )
        print("Hint: Make sure Weaviate is running (make up)", file=sys.stderr)
        sys.exit(1)
