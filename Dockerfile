# Benchmark client image (download, load, index, check, `db` CLI).
# The database runs from the official Weaviate image, see docker-compose.yml.
FROM python:3.12.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY config ./config
COPY src ./src
COPY scripts ./scripts
COPY db /usr/local/bin/db
RUN chmod +x /usr/local/bin/db

CMD ["db", "--help"]
