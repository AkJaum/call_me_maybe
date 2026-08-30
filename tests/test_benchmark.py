"""Tests for reproducible benchmark loading and scoring."""

import json
from pathlib import Path

import pytest

from src.benchmark import BenchmarkCase, build_report, load_cases
from src.io import InputFileError
from src.models import (
    FunctionCallResult,
    FunctionDefinition,
    TypeDefinition,
    build_function_call_result,
)


def make_result(prompt: str, value: float) -> FunctionCallResult:
    """Build one dynamically validated numeric function call."""
    function = FunctionDefinition(
        name="fn_value",
        description="Return a value.",
        parameters={"value": TypeDefinition(type="number")},
        returns=TypeDefinition(type="number"),
    )
    return build_function_call_result(
        prompt, "fn_value", {"value": value}, [function]
    )


def test_report_calculates_rates_and_subject_targets() -> None:
    """Calculate accuracy against labels without feeding labels to generation."""
    cases = [
        BenchmarkCase(
            prompt="one",
            expected_name="fn_value",
            expected_parameters={"value": 1},
        ),
        BenchmarkCase(
            prompt="two",
            expected_name="fn_value",
            expected_parameters={"value": 2},
        ),
    ]
    results = [make_result("one", 1), make_result("two", 2)]
    report = build_report(
        "test-model",
        cases,
        results,
        duration_seconds=10.0,
        peak_memory_mib=100.0,
        failures=[],
    )

    assert report.json_validity == 1.0
    assert report.full_accuracy == 1.0
    assert report.meets_subject_targets


def test_report_counts_failures_and_slow_runs() -> None:
    """Fail targets when generation is invalid, inaccurate, or too slow."""
    cases = [
        BenchmarkCase(
            prompt="one",
            expected_name="fn_value",
            expected_parameters={"value": 1},
        )
    ]
    report = build_report(
        "test-model",
        cases,
        [None],
        duration_seconds=300.0,
        peak_memory_mib=100.0,
        failures=["case 1 failed"],
    )
    assert report.json_validity == 0.0
    assert report.function_accuracy == 0.0
    assert not report.meets_subject_targets


def test_benchmark_cases_are_strictly_validated(tmp_path: Path) -> None:
    """Reject empty cases and labels with unexpected fields."""
    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    with pytest.raises(InputFileError, match="at least one"):
        load_cases(empty)

    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        json.dumps(
            [
                {
                    "prompt": "test",
                    "expected_name": "fn_test",
                    "expected_parameters": {},
                    "extra": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(InputFileError, match="invalid benchmark"):
        load_cases(invalid)
