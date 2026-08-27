"""Tests for mandatory README content from the project subject."""

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
