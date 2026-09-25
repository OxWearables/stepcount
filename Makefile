# Developer convenience targets. Run `make` or `make help` to list them.
# PYTHON can be overridden, e.g. `make test PYTHON=python3.10`.
PYTHON ?= python

.DEFAULT_GOAL := help
.PHONY: help install lint typecheck test test-cov test-fast check clean

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Editable install with dev dependencies
	$(PYTHON) -m pip install -e .[dev]

lint: ## Run Ruff the same way CI does (syntax errors fail; style is advisory)
	$(PYTHON) -m ruff check . --select=E9,F63,F7,F82
	$(PYTHON) -m ruff check . --exit-zero

typecheck: ## Type-check the production package with mypy
	$(PYTHON) -m mypy

test: ## Run the test suite
	$(PYTHON) -m pytest

test-cov: ## Run tests with a coverage report
	$(PYTHON) -m pytest --cov --cov-report=term-missing

test-fast: ## Run tests, skipping ones marked slow
	$(PYTHON) -m pytest -m "not slow"

check: lint typecheck test ## Run all local quality checks

clean: ## Remove build artifacts and caches
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
