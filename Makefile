PYTHON := uv run python
MOULINETTE_DATA := moulinette/successfully
MOULINETTE_OUTPUT := $(MOULINETTE_DATA)/output/function_calling_results.json

.PHONY: install run debug clean lint lint-strict test model-check benchmark \
	visualize moulinette-run moulinette-test

install:
	uv sync

run:
	$(PYTHON) -m src

debug:
	$(PYTHON) -m pdb -m src

model-check:
	$(PYTHON) -m src --inspect-model

benchmark:
	$(PYTHON) -m src.benchmark

visualize:
	$(PYTHON) -m src --visualize data/output/generation_trace.html

moulinette-run:
	$(PYTHON) -m src \
		--functions_definition $(MOULINETTE_DATA)/input/functions_definition.json \
		--input $(MOULINETTE_DATA)/input/function_calling_tests.json \
		--output $(MOULINETTE_OUTPUT)

moulinette-test: moulinette-run
	cd moulinette && uv run python -m moulinette grade_student_answers \
		--set private \
		--student_answer_path successfully/output/function_calling_results.json

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
