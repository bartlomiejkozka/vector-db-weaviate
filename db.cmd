@echo off
REM Usage: db <query_number> [params...]
REM Runs the specified query against Weaviate inside the Docker container.
docker compose exec app ./db %*

