# Thin wrapper around docker compose. Usage: make <target> [DS=<dataset>] [ARGS="--param k=v"]
.PHONY: help build up down download setup ddl load index check show-config db mem-peak status logs test clean

DS ?= $(WVB_DATASET)
COMPOSE = docker compose
RUN = $(COMPOSE) run --rm

# Host info logged with every result
export HOST_CPU_MODEL ?= $(shell sysctl -n machdep.cpu.brand_string 2>/dev/null || grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)
export HOST_OS ?= $(shell (sw_vers -productName && sw_vers -productVersion) 2>/dev/null | xargs || uname -sr)
export HOST_RAM_GB ?= $(shell echo $$(( $$(sysctl -n hw.memsize 2>/dev/null || awk '/MemTotal/{print $$2*1024}' /proc/meminfo) / 1073741824 )))

help: ## Show targets
	@grep -E '^[a-zA-Z_ -]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build the client image
	$(COMPOSE) build

up: ## Start Weaviate (4 cores, 8 GB, swap off)
	$(COMPOSE) up -d --wait weaviate

down: ## Stop all containers (keeps the database volume)
	$(COMPOSE) down

download: ## Download a dataset: make download DS=glove-200-cosine
	$(RUN) download $(DS)

setup: up ## Schema -> load -> index -> check -> build.csv: make setup DS=... [ARGS=...]
	$(RUN) setup $(DS) $(ARGS)

ddl load index check: up ## Run a single step: make load DS=...
	$(RUN) benchmark python scripts/$@.py $(DS) $(ARGS)

show-config: ## Resolved configuration: make show-config DS=... (or DS=--all)
	$(RUN) --no-deps benchmark python scripts/show_config.py $(DS) $(ARGS)

db: ## Run a goal: make db Q="1 0 10" [DS=...]
	$(RUN) benchmark db $(Q) $(if $(DS),--dataset $(DS))

mem-peak: ## Weaviate container memory.peak / memory.current (bytes)
	@$(COMPOSE) exec weaviate sh -c 'echo "peak    $$(cat /sys/fs/cgroup/memory.peak)"; echo "current $$(cat /sys/fs/cgroup/memory.current)"'

status: ## Container status and Weaviate version
	$(COMPOSE) ps
	@curl -sf http://127.0.0.1:8080/v1/meta | python3 -c "import json,sys; print('weaviate', json.load(sys.stdin)['version'])" || echo "Weaviate not reachable"

logs: ## Follow Weaviate logs
	$(COMPOSE) logs -f weaviate

test: ## Unit tests of the configuration logic (local Python >= 3.11)
	PYTHONPATH=src python3 -m unittest discover -s tests -v

clean: ## Remove containers and the database volume (keeps data/ and results/)
	$(COMPOSE) down -v
