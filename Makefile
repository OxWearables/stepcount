# Developer convenience targets. Run `make` or `make help` to list them.
# PYTHON can be overridden, e.g. `make test PYTHON=python3.10`.
PYTHON ?= python

.DEFAULT_GOAL := help
.PHONY: help install lint test test-cov test-fast clean

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Editable install with dev dependencies
	$(PYTHON) -m pip install -e .[dev]

lint: ## Run flake8 the same way CI does (syntax errors fail; style is advisory)
	$(PYTHON) -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	$(PYTHON) -m flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

test: ## Run the test suite
	$(PYTHON) -m pytest

test-cov: ## Run tests with a coverage report
	$(PYTHON) -m pytest --cov --cov-report=term-missing

test-fast: ## Run tests, skipping ones marked slow
	$(PYTHON) -m pytest -m "not slow"

clean: ## Remove build artifacts and caches
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
