PYTHON := uv run python

.PHONY: install run debug clean lint lint-strict test model-check

install:
	uv sync

run:
	$(PYTHON) -m src --validate-only

debug:
	$(PYTHON) -m pdb -m src --validate-only

model-check:
	$(PYTHON) -m src --inspect-model

test:
	uv run pytest

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

clean:
	find . -type d \( -name __pycache__ -o -name .mypy_cache \
		-o -name .pytest_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
