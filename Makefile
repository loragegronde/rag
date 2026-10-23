PYTHON ?= python3
UV ?= uv
VENV_DIR ?= /tmp/rag-venv
VENV_LINK := .venv
UV_PROJECT_ENVIRONMENT := $(VENV_DIR)
UV_CACHE_DIR ?= /tmp/rag-uv-cache
HF_HOME ?= /tmp/rag-hf-cache
# Optional secrets (HF_TOKEN, …) live in gitignored .env — never commit them.
-include .env
export UV_PROJECT_ENVIRONMENT UV_CACHE_DIR HF_HOME HF_TOKEN

LINT_PATHS := src

.PHONY: install venv-link run debug clean fclean re lint lint-strict

install:
	$(UV) sync
	@$(MAKE) --no-print-directory venv-link

venv-link:
	@if [ -e $(VENV_LINK) ] && [ ! -L $(VENV_LINK) ]; then rm -rf $(VENV_LINK); fi
	@ln -sfn $(VENV_DIR) $(VENV_LINK)

run: venv-link
	$(UV) run $(PYTHON) -m src

debug: venv-link
	$(UV) run $(PYTHON) -m pdb -m src

clean:
	find . -path ./data -prune -o -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -path ./data -prune -o -type d -name .mypy_cache -prune -exec rm -rf {} +
	find . -path ./data -prune -o -type f -name "*.py[co]" -exec rm -f {} +

fclean: clean
	rm -rf $(VENV_DIR)
	rm -f $(VENV_LINK)

re: fclean install

lint: venv-link
	$(UV) run flake8 $(LINT_PATHS)
	$(UV) run mypy $(LINT_PATHS) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: venv-link
	$(UV) run flake8 $(LINT_PATHS)
	$(UV) run mypy $(LINT_PATHS) --strict --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
