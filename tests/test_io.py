"""Tests for project input validation and output writing."""

from pathlib import Path

from src.io import load_function_definitions, load_prompts


def test_sample_inputs_are_valid() -> None:
    """Ensure repository demonstration inputs match their Pydantic schemas."""
    definitions = load_function_definitions(
        Path("data/input/functions_definition.json")
    )
    prompts = load_prompts(Path("data/input/function_calling_tests.json"))

    assert definitions[0].name == "fn_add_numbers"
    assert prompts[0].prompt == "What is the sum of 2 and 3?"
