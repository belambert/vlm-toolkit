.PHONY: check format

check:
	@echo "Running style checks..."
	black --check --diff src/
	isort --check-only --diff src/

format:
	@echo "Formatting code..."
	black src/
	isort src/
