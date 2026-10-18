.PHONY: help setup up down load query clean logs status

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Build and start all containers
	docker compose build
	docker compose up -d
	@echo ""
	@echo "Waiting for Weaviate to be ready..."
	@docker compose exec app bash -c 'until curl -sf http://weaviate:8080/v1/.well-known/ready > /dev/null 2>&1; do sleep 1; done'
	@echo "Weaviate is ready!"
	@echo ""
	@echo "Run 'make load' to download and import the dataset."

up: ## Start containers
	docker compose up -d

down: ## Stop containers
	docker compose down

load: ## Download dataset and import into Weaviate
	docker compose exec app python -m src.load_data

query: ## Run a query (usage: make query Q="1 100 5")
	@if [ -z "$(Q)" ]; then \
		echo "Usage: make query Q=\"<query_number> [params...]\""; \
		echo "Example: make query Q=\"1 100 5\""; \
		exit 1; \
	fi
	docker compose exec app ./db $(Q)

db: ## Alias: run a query (usage: make db Q="1 100 5")
	@$(MAKE) query Q="$(Q)"

status: ## Show container status and Weaviate info
	docker compose ps
	@echo ""
	@docker compose exec app curl -sf http://weaviate:8080/v1/meta | python -m json.tool 2>/dev/null || echo "Weaviate not reachable"

logs: ## Show container logs
	docker compose logs -f

clean: ## Stop containers, remove volumes and downloaded data
	docker compose down -v
	@echo "Cleaned up containers and volumes."

reset: ## Full reset: clean + rebuild
	$(MAKE) clean
	$(MAKE) setup
