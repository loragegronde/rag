PYTHON ?= python3
UV ?= uv
UV_PROJECT_ENVIRONMENT ?= /tmp/rag-venv
UV_CACHE_DIR ?= /tmp/rag-uv-cache
HF_HOME ?= /tmp/rag-hf-cache
export UV_PROJECT_ENVIRONMENT UV_CACHE_DIR HF_HOME

.PHONY: install run debug clean lint lint-strict

install:
	$(UV) sync

run:
	$(UV) run $(PYTHON) -m src

debug:
	$(UV) run $(PYTHON) -m pdb -m src

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .mypy_cache -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

lint:
	$(UV) run flake8 . --exclude .venv
	$(UV) run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude .venv

lint-strict:
	$(UV) run flake8 . --exclude .venv
	$(UV) run mypy . --strict --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude .venv