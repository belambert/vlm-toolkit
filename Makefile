.PHONY: check format test install install-ml

check:
	@echo "Running style checks..."
	black --check --diff src/
	isort --check-only --diff src/

format:
	@echo "Formatting code..."
	black src/
	isort src/

test:
	@echo "Installing dependencies..."
	@uv sync --extra dev --quiet
	@echo "Running tests..."
	@uv run pytest

install:
	@echo "Installing dependencies..."
	uv sync --extra ml
