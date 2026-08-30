"""Tests for the complete command-line application pipeline."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.__main__ import main
from src.generation import ConstrainedDecoder, GenerationError
from src.model import QwenClient
from src.models import build_function_call_result


def write_inputs(tmp_path: Path, prompts: list[str]) -> tuple[Path, Path]:
    """Write a valid function catalog and an arbitrary prompt batch."""
    definitions = tmp_path / "functions.json"
    requests = tmp_path / "prompts.json"
    definitions.write_text(
        json.dumps(
            [
                {
                    "name": "fn_echo",
                    "description": "Echo text.",
                    "parameters": {"text": {"type": "string"}},
                    "returns": {"type": "string"},
                }
            ]
        ),
        encoding="utf-8",
    )
    requests.write_text(
        json.dumps([{"prompt": prompt} for prompt in prompts]),
        encoding="utf-8",
    )
    return definitions, requests


def test_cli_processes_all_prompts_and_writes_custom_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reuse one decoder, preserve order, and emit the exact output schema."""
    definitions_path, prompts_path = write_inputs(
        tmp_path, ["Echo one", "Echo two"]
    )
    output = tmp_path / "out" / "calls.json"
    loaded: list[QwenClient] = []

    def fake_load(client: QwenClient) -> None:
        """Record that the model is initialized exactly once."""
        loaded.append(client)

    monkeypatch.setattr(QwenClient, "load", fake_load)
    decoder = MagicMock(spec=ConstrainedDecoder)

    def fake_from_client(client: QwenClient) -> MagicMock:
        """Return one shared decoder after model initialization."""
        assert client is loaded[0]
        return decoder

    monkeypatch.setattr(ConstrainedDecoder, "from_client", fake_from_client)
    from src.io import load_function_definitions

    functions = load_function_definitions(definitions_path)
    decoder.generate.side_effect = [
        build_function_call_result(
            "Echo one", "fn_echo", {"text": "one"}, functions
        ),
        build_function_call_result(
            "Echo two", "fn_echo", {"text": "two"}, functions
        ),
    ]

    status = main(
        [
            "--functions_definition",
            str(definitions_path),
            "--input",
            str(prompts_path),
            "--output",
            str(output),
        ]
    )

    assert status == 0
    assert len(loaded) == 1
    assert decoder.generate.call_count == 2
    assert json.loads(output.read_text(encoding="utf-8")) == [
        {
            "prompt": "Echo one",
            "name": "fn_echo",
            "parameters": {"text": "one"},
        },
        {
            "prompt": "Echo two",
            "name": "fn_echo",
            "parameters": {"text": "two"},
        },
    ]
    assert f"Wrote 2 function calls to {output}." in capsys.readouterr().out


def test_cli_generation_failure_does_not_overwrite_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Return a readable error and preserve prior output on partial batch failure."""
    definitions, prompts = write_inputs(tmp_path, ["first", "second"])
    output = tmp_path / "calls.json"
    output.write_text('{"previous":true}\n', encoding="utf-8")
    monkeypatch.setattr(QwenClient, "load", lambda self: None)
    decoder = MagicMock(spec=ConstrainedDecoder)
    decoder.generate.side_effect = GenerationError("no valid continuation")
    monkeypatch.setattr(
        ConstrainedDecoder, "from_client", lambda client: decoder
    )

    status = main(
        [
            "--functions_definition",
            str(definitions),
            "--input",
            str(prompts),
            "--output",
            str(output),
        ]
    )

    assert status == 1
    assert output.read_text(encoding="utf-8") == '{"previous":true}\n'
    assert "error: no valid continuation" in capsys.readouterr().err


def test_cli_empty_batch_writes_array_without_loading_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handle an empty test array efficiently and deterministically."""
    definitions, prompts = write_inputs(tmp_path, [])
    output = tmp_path / "empty.json"

    def forbidden_load(client: QwenClient) -> None:
        """Fail if the empty-batch fast path initializes Qwen."""
        raise AssertionError(f"model should not load: {client}")

    monkeypatch.setattr(QwenClient, "load", forbidden_load)
    status = main(
        [
            "--functions_definition",
            str(definitions),
            "--input",
            str(prompts),
            "--output",
            str(output),
        ]
    )

    assert status == 0
    assert json.loads(output.read_text(encoding="utf-8")) == []


def test_cli_empty_batch_can_write_visualization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create the optional report without loading Qwen for an empty batch."""
    definitions, prompts = write_inputs(tmp_path, [])
    output = tmp_path / "empty.json"
    report = tmp_path / "trace.html"

    def forbidden_load(client: QwenClient) -> None:
        """Fail if optional empty reporting initializes the model."""
        raise AssertionError(f"model should not load: {client}")

    monkeypatch.setattr(QwenClient, "load", forbidden_load)
    status = main(
        [
            "--functions_definition",
            str(definitions),
            "--input",
            str(prompts),
            "--output",
            str(output),
            "--visualize",
            str(report),
        ]
    )

    assert status == 0
    assert "No prompts were provided." in report.read_text(encoding="utf-8")


def test_cli_invalid_input_fails_before_loading_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Validate files before performing expensive model initialization."""
    definitions, _ = write_inputs(tmp_path, ["unused"])
    missing = tmp_path / "missing.json"

    def forbidden_load(client: QwenClient) -> None:
        """Fail if invalid input reaches model initialization."""
        raise AssertionError(f"model should not load: {client}")

    monkeypatch.setattr(QwenClient, "load", forbidden_load)
    status = main(
        [
            "--functions_definition",
            str(definitions),
            "--input",
            str(missing),
        ]
    )

    assert status == 1
    assert "input file not found" in capsys.readouterr().err
