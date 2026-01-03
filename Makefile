.PHONY: check format test

check:
	@echo "Running style checks..."
	black --check --diff src/
	isort --check-only --diff src/

format:
	@echo "Formatting code..."
	black src/
	isort src/

test:
	@echo "Running tests..."
	uv sync --extra dev
	uv run pytest
