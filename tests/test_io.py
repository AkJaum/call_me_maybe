"""Tests for project input validation and output writing."""

from pathlib import Path

import json

import pytest

from src.io import (
    InputFileError,
    load_function_definitions,
    load_prompts,
    write_results,
)
from src.models import build_function_call_result


def test_repository_inputs_follow_the_dynamic_schemas() -> None:
    """Accept any repository inputs that satisfy the subject's schemas."""
    definitions = load_function_definitions(
        Path("data/input/functions_definition.json")
    )
    prompts = load_prompts(Path("data/input/function_calling_tests.json"))

    assert definitions
    assert isinstance(prompts, list)


def test_invalid_json_has_a_readable_error(tmp_path: Path) -> None:
    """Translate malformed JSON into the project's public file error."""
    path = tmp_path / "broken.json"
    path.write_text("[}", encoding="utf-8")
    with pytest.raises(InputFileError, match="invalid JSON"):
        load_prompts(path)


def test_invalid_utf8_has_a_readable_error(tmp_path: Path) -> None:
    """Translate a non-text input file into a specific controlled error."""
    path = tmp_path / "binary.json"
    path.write_bytes(b'[{"prompt":"\xff"}]')

    with pytest.raises(InputFileError, match="not valid UTF-8"):
        load_prompts(path)


def test_missing_input_file_has_a_readable_error(tmp_path: Path) -> None:
    """Translate a missing file into the project's public file error."""
    with pytest.raises(InputFileError, match="input file not found"):
        load_prompts(tmp_path / "missing.json")


@pytest.mark.parametrize(
    "document",
    [
        {},
        [{}],
        [{"prompt": ""}],
        [{"prompt": "ok", "extra": True}],
    ],
)
def test_invalid_prompt_documents_are_rejected(
    tmp_path: Path, document: object
) -> None:
    """Reject wrong roots, missing/empty prompts, and unexpected fields."""
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(InputFileError, match="invalid prompts"):
        load_prompts(path)


def test_duplicate_function_names_are_rejected(tmp_path: Path) -> None:
    """Reject an ambiguous function catalog at the input boundary."""
    function = {
        "name": "fn_same",
        "description": "Same name.",
        "parameters": {},
        "returns": {"type": "string"},
    }
    path = tmp_path / "functions.json"
    path.write_text(json.dumps([function, function]), encoding="utf-8")
    with pytest.raises(InputFileError, match="unique"):
        load_function_definitions(path)


def test_results_are_written_with_exact_keys_and_unicode(tmp_path: Path) -> None:
    """Write the required JSON array and preserve non-ASCII arguments."""
    function_document = {
        "name": "fn_greet",
        "description": "Greet a person.",
        "parameters": {"name": {"type": "string"}},
        "returns": {"type": "string"},
    }
    functions_path = tmp_path / "functions.json"
    functions_path.write_text(json.dumps([function_document]), encoding="utf-8")
    functions = load_function_definitions(functions_path)
    result = build_function_call_result(
        "Cumprimente José", "fn_greet", {"name": "José"}, functions
    )
    output = tmp_path / "nested" / "results.json"

    write_results(output, [result])

    assert json.loads(output.read_text(encoding="utf-8")) == [
        {
            "prompt": "Cumprimente José",
            "name": "fn_greet",
            "parameters": {"name": "José"},
        }
    ]
    assert "José" in output.read_text(encoding="utf-8")


def test_failed_atomic_write_preserves_previous_target(tmp_path: Path) -> None:
    """Remove temporary output and preserve the target when replacement fails."""
    output = tmp_path / "results.json"
    output.mkdir()
    marker = output / "marker"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(InputFileError, match="could not write"):
        write_results(output, [])

    assert marker.read_text(encoding="utf-8") == "keep"
    assert not list(tmp_path.glob(".results.json.*.tmp"))
