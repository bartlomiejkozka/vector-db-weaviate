"""Weaviate connection helpers."""

from __future__ import annotations

import time

import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout

from wvbench import settings


def connect(wait_s: float = 120.0) -> weaviate.WeaviateClient:
    """Connect over HTTP + gRPC, waiting up to wait_s seconds for the server to become ready."""
    deadline = time.monotonic() + wait_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            client = weaviate.connect_to_custom(
                http_host=settings.WEAVIATE_HOST,
                http_port=settings.WEAVIATE_HTTP_PORT,
                http_secure=False,
                grpc_host=settings.WEAVIATE_HOST,
                grpc_port=settings.WEAVIATE_GRPC_PORT,
                grpc_secure=False,
                additional_config=AdditionalConfig(timeout=Timeout(init=10, query=120, insert=600)),
            )
            if client.is_ready():
                return client
            client.close()
        except Exception as e:  # server still starting
            last_error = e
        time.sleep(1.0)
    raise ConnectionError(
        f"Weaviate not ready at {settings.WEAVIATE_HOST}:{settings.WEAVIATE_HTTP_PORT} "
        f"after {wait_s:.0f}s: {last_error}"
    )


def server_version(client: weaviate.WeaviateClient) -> str:
    return str(client.get_meta().get("version", "unknown"))
