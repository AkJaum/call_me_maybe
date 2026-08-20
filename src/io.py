"""JSON loading, validation, and output helpers."""

import json
from pathlib import Path
from pydantic import TypeAdapter, ValidationError

from src.models import FunctionCallResult, FunctionDefinition, PromptInput


class InputFileError(ValueError):
    """Report a readable problem with a project input file."""


def _load_json(path: Path) -> object:
    """Load a JSON document and translate filesystem errors."""
    try:
        with path.open(encoding="utf-8") as stream:
            raw_data: object = json.load(stream)
        return raw_data
    except FileNotFoundError as exc:
        raise InputFileError(f"input file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InputFileError(
            f"invalid JSON in {path} at line {exc.lineno}, column {exc.colno}"
        ) from exc
    except OSError as exc:
        raise InputFileError(f"could not read {path}: {exc}") from exc


def load_function_definitions(path: Path) -> list[FunctionDefinition]:
    """Load available function definitions."""
    try:
        return TypeAdapter(list[FunctionDefinition]).validate_python(_load_json(path))
    except ValidationError as exc:
        raise InputFileError(f"invalid function definitions in {path}: {exc}") from exc


def load_prompts(path: Path) -> list[PromptInput]:
    """Load natural-language prompts."""
    try:
        return TypeAdapter(list[PromptInput]).validate_python(_load_json(path))
    except ValidationError as exc:
        raise InputFileError(f"invalid prompts in {path}: {exc}") from exc


def write_results(path: Path, results: list[FunctionCallResult]) -> None:
    """Write validated results as formatted JSON, creating the parent directory."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [result.model_dump(mode="json") for result in results]
        with path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    except OSError as exc:
        raise InputFileError(f"could not write output file {path}: {exc}") from exc
