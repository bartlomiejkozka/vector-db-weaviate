"""Weaviate client connection helper."""

import os
import sys
import weaviate


def get_client() -> weaviate.WeaviateClient:
    """Create and return a connected Weaviate client.
    
    Reads connection parameters from environment variables:
    - WEAVIATE_URL (default: http://localhost:8080)
    - WEAVIATE_GRPC_URL (default: localhost:50051)
    """
    url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    grpc_host = os.getenv("WEAVIATE_GRPC_URL", "localhost:50051")
    
    # Parse host and port from WEAVIATE_GRPC_URL
    grpc_parts = grpc_host.split(":")
    grpc_hostname = grpc_parts[0]
    grpc_port = int(grpc_parts[1]) if len(grpc_parts) > 1 else 50051
    
    # Parse host and port from WEAVIATE_URL  
    from urllib.parse import urlparse
    parsed = urlparse(url)
    http_host = parsed.hostname or "localhost"
    http_port = parsed.port or 8080
    
    try:
        client = weaviate.connect_to_custom(
            http_host=http_host,
            http_port=http_port,
            http_secure=False,
            grpc_host=grpc_hostname,
            grpc_port=grpc_port,
            grpc_secure=False,
        )
        return client
    except Exception as e:
        print(f"Error: Could not connect to Weaviate at {url}: {e}", file=sys.stderr)
        print("Hint: Make sure Weaviate is running (docker compose up -d)", file=sys.stderr)
        sys.exit(1)
