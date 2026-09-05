.PHONY: check format test install

check:
	@echo "Running style checks..."
	uv run black --check --diff src/
	uv run isort --check-only --diff src/
	uv run mypy

format:
	@echo "Formatting code..."
	uv run black src/
	uv run isort src/

test:
	@echo "Running tests..."
	@uv run pytest

install:
	@echo "Installing dependencies..."
	uv sync --extra dev
