"""Tests for mandatory README content from the project subject."""

import ast
from pathlib import Path


def test_readme_contains_every_mandatory_section() -> None:
    """Keep the required English documentation present during later edits."""
    readme = Path("README.md").read_text(encoding="utf-8")
    assert readme.startswith(
        "*This project has been created as part of the 42 curriculum by akjaum.*"
    )
    required_headings = [
        "## Description",
        "## Instructions",
        "## Resources",
        "## Use of AI",
        "## Constrained decoding algorithm",
        "## Design decisions",
        "## Performance analysis",
        "## Challenges faced",
        "## Testing strategy",
    ]
    for heading in required_headings:
        assert heading in readme


def test_readme_documents_required_command_and_output_keys() -> None:
    """Document the evaluator command and the subject's exact output fields."""
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "uv run python -m src" in readme
    assert "`prompt`, `name`, and `parameters`" in readme


def test_student_code_does_not_import_forbidden_model_frameworks() -> None:
    """Keep model-framework internals behind the supplied llm_sdk boundary."""
    forbidden = {
        "accelerate",
        "dspy",
        "huggingface_hub",
        "outlines",
        "torch",
        "transformers",
    }
    imported: set[str] = set()
    for path in Path("src").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module.split(".", 1)[0])

    assert imported.isdisjoint(forbidden)
