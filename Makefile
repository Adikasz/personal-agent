# =============================================================================
# Developer convenience targets.
# Requires `make` (available on Linux/macOS natively and on Windows via
# Git Bash or `choco install make`).
# =============================================================================

PYTHON ?= python
PIP    ?= $(PYTHON) -m pip

.PHONY: help install install-dev lint format typecheck test coverage ci clean

help: ## Show this help message.
	@echo "Available targets:"
	@echo "  install       Install runtime dependencies only"
	@echo "  install-dev   Install runtime + development dependencies"
	@echo "  lint          Run ruff linter and format check"
	@echo "  format        Auto-format code with ruff"
	@echo "  typecheck     Run mypy in strict mode"
	@echo "  test          Run pytest verbosely"
	@echo "  coverage      Run tests with branch coverage and HTML report"
	@echo "  ci            Run the full CI pipeline locally (lint + typecheck + test)"
	@echo "  clean         Remove build, cache and coverage artifacts"

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements-dev.txt

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	mypy .

test:
	pytest -v

coverage:
	pytest --cov=core --cov=utils --cov=agents --cov-branch \
	       --cov-report=term-missing --cov-report=html

ci: lint typecheck test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
