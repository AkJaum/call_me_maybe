"""Tests for the optional constrained-generation HTML report."""

from pathlib import Path

import pytest

from src.generation import GenerationStep, GenerationTrace
from src.models import FunctionDefinition, TypeDefinition, build_function_call_result
from src.visualization import (
    VisualizationError,
    render_generation_report,
    write_generation_report,
)


def make_trace(prompt: str = "Add 2 and 3") -> GenerationTrace:
    """Build a validated trace fixture without loading the real model."""
    function = FunctionDefinition(
        name="fn_add",
        description="Add two numbers.",
        parameters={
            "a": TypeDefinition(type="number"),
            "b": TypeDefinition(type="number"),
        },
        returns=TypeDefinition(type="number"),
    )
    result = build_function_call_result(
        prompt, "fn_add", {"a": 2, "b": 3}, [function]
    )
    return GenerationTrace(
        prompt=prompt,
        model_name="Qwen/Qwen3-0.6B",
        duration_seconds=1.25,
        cache_hits=2,
        cache_misses=1,
        steps=(
            GenerationStep(
                index=1,
                token_id=90,
                token_fragment="{",
                allowed_token_count=3,
                selected_logit=4.5,
                prefix="{",
            ),
        ),
        result=result,
    )


def test_report_visualizes_decisions_and_escapes_untrusted_text() -> None:
    """Expose useful trace fields without allowing prompt HTML injection."""
    report = render_generation_report([make_trace("<script>alert(1)</script>")])

    assert "Constrained decoding trace" in report
    assert "Qwen/Qwen3-0.6B" in report
    assert "90" in report
    assert "4.5" in report
    assert "2 hits / 1 misses" in report
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in report
    assert "<h2><script>" not in report


def test_report_is_written_atomically(tmp_path: Path) -> None:
    """Write a complete standalone document to a newly created directory."""
    output = tmp_path / "nested" / "trace.html"

    write_generation_report(output, [make_trace()])

    document = output.read_text(encoding="utf-8")
    assert document.startswith("<!doctype html>")
    assert "Validated result" in document
    assert not list(output.parent.glob(".trace.html.*.tmp"))


def test_failed_report_write_preserves_previous_target(tmp_path: Path) -> None:
    """Translate filesystem failures and remove the temporary trace file."""
    output = tmp_path / "trace.html"
    output.mkdir()
    marker = output / "marker"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(VisualizationError, match="could not write visualization"):
        write_generation_report(output, [make_trace()])

    assert marker.read_text(encoding="utf-8") == "keep"
    assert not list(tmp_path.glob(".trace.html.*.tmp"))
