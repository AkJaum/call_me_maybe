"""Tests for dynamic validation against function definitions."""

import math

import pytest
from pydantic import ValidationError

from src.models import (
    FunctionCatalog,
    FunctionDefinition,
    TypeDefinition,
    build_function_call_result,
)


def make_function(
    name: str = "fn_example", **parameters: TypeDefinition
) -> FunctionDefinition:
    """Create a concise function definition for schema validation tests."""
    return FunctionDefinition(
        name=name,
        description="Test function.",
        parameters=parameters,
        returns=TypeDefinition(type="string"),
    )


def test_catalog_rejects_empty_and_duplicate_functions() -> None:
    """Ensure a catalog always provides unambiguous choices to the model."""
    with pytest.raises(ValidationError, match="at least one"):
        FunctionCatalog.model_validate([])

    function = make_function()
    with pytest.raises(ValidationError, match="unique"):
        FunctionCatalog.model_validate([function, function])


def test_complex_argument_types_are_explicitly_rejected() -> None:
    """Keep nested array/object support out of the mandatory scalar contract."""
    for unsupported in ("array", "object"):
        with pytest.raises(ValidationError):
            TypeDefinition.model_validate({"type": unsupported})


def test_result_requires_exact_name_keys_and_types() -> None:
    """Validate every generated field against the selected function schema."""
    function = make_function(
        text=TypeDefinition(type="string"),
        count=TypeDefinition(type="integer"),
        ratio=TypeDefinition(type="number"),
        enabled=TypeDefinition(type="boolean"),
    )
    result = build_function_call_result(
        "Run it",
        "fn_example",
        {"text": "ok", "count": 2, "ratio": 0.5, "enabled": True},
        [function],
    )
    assert result.name == "fn_example"

    invalid_cases = [
        ("missing parameters", {"text": "ok"}),
        (
            "extra parameters",
            {"text": "ok", "count": 2, "ratio": 1, "enabled": True, "x": 1},
        ),
        (
            "must have type integer",
            {"text": "ok", "count": True, "ratio": 1, "enabled": True},
        ),
        (
            "must have type boolean",
            {"text": "ok", "count": 2, "ratio": 1, "enabled": 1},
        ),
    ]
    for message, parameters in invalid_cases:
        with pytest.raises(ValidationError, match=message):
            build_function_call_result(
                "Run it", "fn_example", parameters, [function]  # type: ignore[arg-type]
            )

    with pytest.raises(ValidationError, match="unknown function"):
        build_function_call_result("Run it", "fn_missing", {}, [function])


def test_number_rejects_non_finite_values() -> None:
    """Prevent values that Python can hold but JSON cannot represent."""
    function = make_function(value=TypeDefinition(type="number"))
    for value in (math.inf, -math.inf, math.nan):
        with pytest.raises(ValidationError, match="must have type number"):
            build_function_call_result(
                "Run it", function.name, {"value": value}, [function]
            )
