.PHONY: help setup venv up down load query db clean logs status reset

PYTHON ?= python3
VENV = .venv
VENV_PY = $(VENV)/bin/python

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: venv up ## Create virtualenv, install dependencies and start Weaviate
	@echo "Waiting for Weaviate to be ready..."
	@until curl -sf http://localhost:8080/v1/.well-known/ready > /dev/null 2>&1; do sleep 1; done
	@echo "Weaviate is ready!"
	@echo ""
	@echo "Run 'make load' to download and import the dataset."

venv: ## Create virtualenv and install Python dependencies
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	$(VENV_PY) -m pip install -q -r requirements.txt

up: ## Start Weaviate container
	docker compose up -d

down: ## Stop Weaviate container
	docker compose down

load: ## Download dataset and import into Weaviate
	$(VENV_PY) -m src.load_data

query: ## Run a query (usage: make query Q="1 100 5")
	@if [ -z "$(Q)" ]; then \
		echo "Usage: make query Q=\"<query_number> [params...]\""; \
		echo "Example: make query Q=\"1 100 5\""; \
		exit 1; \
	fi
	./db $(Q)

db: query ## Alias for query (usage: make db Q="1 100 5")

status: ## Show container status and Weaviate info
	docker compose ps
	@echo ""
	@curl -sf http://localhost:8080/v1/meta | $(PYTHON) -m json.tool 2>/dev/null || echo "Weaviate not reachable"

logs: ## Show Weaviate logs
	docker compose logs -f

clean: ## Stop Weaviate, remove its volume and downloaded data
	docker compose down -v
	rm -rf data
	@echo "Cleaned up container, volume and downloaded data."

reset: ## Full reset: clean + setup
	$(MAKE) clean
	$(MAKE) setup
